# Lab 6.9 (Julia) — 세 상 혼합 XRD 패턴의 PCA·ICA·NMF 분리 (합성 데이터: 실제 상이 아님)
# 파이썬 labs/64_unmix.py 대응. PCA는 SVD, ICA는 sklearn FastICA(whiten='unit-variance', 대칭 갱신, g = tanh)와
# 같은 알고리즘을 직접 구현했고(MultivariateStats.jl의 ICA도 같은 FastICA지만 키워드 이름이 판마다 달라
# 짧은 구현을 택했다), NMF는 NMF.jl(nndsvda 초기화, 좌표하강), NNLS는 NonNegLeastSquares.jl,
# 헝가리안 매칭은 3! = 6개 순열을 모두 보는 것으로 대신한다.
# 참 스펙트럼의 상관·첨도는 결정론적이라 그대로 일치하고, 혼합 분율·잡음·초기값이 Julia RNG를 쓰는 나머지는
# 통계적으로 일치한다.
using LinearAlgebra, Printf, Random, Statistics, StatsBase, Distributions, NMF, NonNegLeastSquares

const PERMS3 = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
const TT = collect(range(10, 70; length = 600))
pv(x, x0; fwhm = 0.4, eta = 0.5) = eta / (1 + 4 * (x - x0)^2 / fwhm^2) + (1 - eta) * exp(-4 * log(2) * (x - x0)^2 / fwhm^2)
phase(peaks) = (s = sum(h .* pv.(TT, x0) for (x0, h) in peaks); s ./ maximum(s))

const PEAKS = [
    ("separated", [[(20.0, 1.0), (28.5, 0.6), (33.0, 0.8), (47.0, 0.4), (56.0, 0.3)],
                   [(22.0, 0.7), (31.0, 1.0), (38.5, 0.5), (44.5, 0.6), (64.0, 0.3)],
                   [(25.5, 1.0), (35.0, 0.4), (41.0, 0.7), (52.0, 0.5), (60.0, 0.4)]]),
    ("overlapping", [[(20.0, 1.0), (28.5, 0.6), (33.0, 0.8), (47.0, 0.4), (56.0, 0.3)],
                     [(22.0, 0.7), (31.0, 1.0), (38.5, 0.5), (44.5, 0.6), (64.0, 0.3)],
                     [(25.5, 1.0), (31.2, 0.8), (41.0, 0.7), (44.7, 0.6), (60.0, 0.4)]]),
]

sym_decorrelate(W) = (F = svd(W); F.U * F.Vt)                 # (W W')^(-1/2) W

"FastICA(대칭, tanh). Xobs: 관측 x 특징. 반환: 특징 x k 분리 행렬(중심화 안 한 자료에 곱해도 된다)."
function fastica(Xobs, k; rng, maxiter = 2000, tol = 1e-6)
    n = size(Xobs, 1); Xc = Xobs .- mean(Xobs; dims = 1)
    F = svd(Xc); K = F.V[:, 1:k] ./ F.S[1:k]' .* sqrt(n)     # 백색화: Xc*K의 열은 분산 1, 서로 무상관
    Xw = Xc * K
    W = sym_decorrelate(randn(rng, k, k))
    for _ in 1:maxiter
        G = tanh.(Xw * W'); Gp = 1 .- G .^ 2
        Wn = sym_decorrelate((G' * Xw) ./ n .- Diagonal(vec(mean(Gp; dims = 1))) * W)
        lim = maximum(abs.(abs.(diag(Wn * W')) .- 1)); W = Wn
        lim < tol && break
    end
    return K * W'
end

"한 단위 FastICA (파이썬 판의 처음부터 구현과 같은 갱신식)"
function fastica_one_unit(Xw, rng; n_iter = 200, tol = 1e-10)
    w = normalize(randn(rng, size(Xw, 2)))
    for it in 1:n_iter
        u = Xw * w; g = tanh.(u); gp = 1 .- g .^ 2
        wn = normalize(vec(mean(Xw .* g; dims = 1)) .- mean(gp) .* w)
        1 - abs(dot(wn, w)) < tol && return wn, it
        w = wn
    end
    return w, n_iter
end

"|상관|의 합이 최대인 순열로 짝짓고 부호를 맞춘다"
function match(S, Se)
    C = [cor(S[i, :], Se[j, :]) for i in 1:3, j in 1:3]
    best = argmax(pm -> sum(abs(C[i, pm[i]]) for i in 1:3), PERMS3)
    est = permutedims(reduce(hcat, [sign(C[i, best[i]]) .* Se[best[i], :] for i in 1:3]))
    return [abs(C[i, best[i]]) for i in 1:3], est
end

function pca_components(X, k)
    Xc = X .- mean(X; dims = 1); F = svd(Xc); Vt = permutedims(F.V[:, 1:k])
    for r in 1:k
        j = argmax(abs.(Vt[r, :])); Vt[r, :] .*= sign(Vt[r, j])
    end
    ev = F.S .^ 2; return Vt, ev ./ sum(ev)
end

function main()
    for (case, peaks) in PEAKS
        rng = MersenneTwister(0)
        S = permutedims(reduce(hcat, [phase(pk) for pk in peaks]))           # 3 x 600 참 스펙트럼
        Fr = permutedims(rand(rng, Dirichlet(3, 1.0), 150))                  # 150 x 3 분율 (합 1)
        X = Fr * S + 0.01 .* randn(rng, 150, 600); Xnn = max.(X, 0.0)
        println("\n=== case: $case peaks ===")
        println("true spectra: pairwise corr = " * join([@sprintf("%+.3f", cor(S[i, :], S[j, :])) for (i, j) in ((1, 2), (1, 3), (2, 3))], ", "))
        println("excess kurtosis of true spectra (over 2-theta channels): " * join([@sprintf("%.1f", kurtosis(S[i, :])) for i in 1:3], ", "))
        println("rank check: singular values of uncentered X (top 4) = " * join([@sprintf("%.2f", v) for v in svdvals(X)[1:4]], ", "))
        Vp, evr = pca_components(X, 3)
        println("PCA explained variance of first 4 PCs: " * join([@sprintf("%.4f", v) for v in evr[1:4]], ", "))
        U = fastica(permutedims(X), 3; rng = MersenneTwister(0))              # 관측 = 2theta 채널
        S_ica = permutedims(permutedims(X) * U)                               # 중심화 안 한 자료에 적용: 기준선 유지
        r = nnmf(Xnn, 3; init = :nndsvda, alg = :cd, maxiter = 5000, tol = 1e-6)
        S_nmf = r.H
        for (name, Se) in (("PCA", Vp), ("ICA", S_ica), ("NMF", S_nmf))
            cc, Sm = match(S, Se)
            neg = [max(-sum(s[s .< 0]) / sum(abs, s), 0.0) for s in eachrow(Sm)]
            line = name * ": |corr| with truth = " * join([@sprintf("%.3f", v) for v in cc], ", ") *
                   ";  negative-intensity share = " * join([@sprintf("%.2f", v) for v in neg], ", ")
            if name == "PCA"
                line *= ";  fraction MAE = n/a"                              # 주성분은 스펙트럼이 아니다
            else
                Fh = permutedims(reduce(hcat, [vec(nonneg_lsq(permutedims(Sm), X[i, :]; alg = :nnls)) for i in 1:150]))
                Fh ./= sum(Fh; dims = 2)
                line *= @sprintf(";  fraction MAE = %.3f", mean(abs.(Fh - Fr)))
            end
            println(line)
        end
        if case == "separated"
            Xc = permutedims(X) .- mean(permutedims(X); dims = 1); F = svd(Xc)
            Xw = F.U[:, 1:3] .* sqrt(size(Xc, 1))
            w, it = fastica_one_unit(Xw, MersenneTwister(3)); u = Xw * w
            cbest = maximum(abs(cor(u, S[i, :])) for i in 1:3)
            @printf("from-scratch one-unit FastICA (tanh): converged in %d iterations; best |corr| with a true spectrum = %.3f; excess kurtosis of u = %.1f\n",
                    it, cbest, kurtosis(u))
            for t_ in (0.0, 0.5, 1.0)
                mix = (1 - t_) .* u .+ t_ .* Xw[:, 1]
                @printf("   kurtosis along (1-t)*w + t*PC1 direction, t = %s: %.2f\n", t_, kurtosis(mix ./ std(mix)))
            end
        end
    end
end

main()
