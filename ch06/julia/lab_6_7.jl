# Lab 6.7 (Julia) — Magpie 특징 공간의 주성분
# 파이썬 labs/64_pca.py 대응. PCA는 중심화한 자료의 SVD로 직접 계산하고(부호는 sklearn 1.5 이후의
# svd_flip 규칙: 각 주성분 벡터에서 절댓값이 가장 큰 성분을 양수로), MultivariateStats.jl의 PCA로
# 고윳값을 대조한다. Part A, B는 결정론적이라 숫자와 부호가 그대로 일치해야 하고, Part C, D의 그룹 5겹은
# 동률 그룹 순서 차이로 겹 구성이 조금 달라 소수 둘째 자리 근처까지 일치한다.
using LinearAlgebra, Printf, Random, Statistics, StatsBase, MultivariateStats
include("ch6_common.jl")

"PCA: (주성분 행렬 Vt(행 = 성분), 분산, 설명 분산비). 표본분산은 n-1로 나눈다(sklearn과 같음)."
function pca_svd(X)
    Xc = X .- mean(X; dims = 1); F = svd(Xc)
    Vt = permutedims(F.V)
    for k in 1:size(Vt, 1)                                   # svd_flip(u_based_decision=False)
        j = argmax(abs.(Vt[k, :])); Vt[k, :] .*= sign(Vt[k, j])
    end
    ev = F.S .^ 2 ./ (size(X, 1) - 1)
    return Vt, ev, ev ./ sum(ev)
end

"표준화 -> PCA(k) -> 선형회귀 (모두 학습 겹에서만 맞춘다)"
function pcr_predict(Xtr, ytr, Xte, k)
    sc = scaler(Xtr); Ztr, Zte = zscore_with(Xtr, sc), zscore_with(Xte, sc)
    mz = mean(Ztr; dims = 1); Vt, _, _ = pca_svd(Ztr); W = permutedims(Vt[1:k, :])
    Ttr, Tte = (Ztr .- mz) * W, (Zte .- mz) * W
    b = hcat(Ttr, ones(size(Ttr, 1))) \ ytr
    return hcat(Tte, ones(size(Tte, 1))) * b
end

function main()
    t0 = time()
    d, X, y, names = load_magpie()
    keep = keep_mask(X; thr = 0.99); X = X[:, keep]; names = names[keep]
    n, p = size(X); metal = y .== 0
    @printf("all compositions: n = %d, p = %d (constant / >99%%-identical columns removed); metals: %d\n", n, p, sum(metal))

    println("\n=== Part A: raw vs standardized ===")
    Vr, _, evr_raw = pca_svd(X); j = argmax(abs.(Vr[1, :]))
    @printf("raw features: PC1 explains %.3f of variance; largest |loading| = %s (%+.3f)\n", evr_raw[1], names[j], Vr[1, j])
    sdv = vec(std(X; dims = 1, corrected = false)); o = sortperm(-sdv)[1:3]
    println("three largest raw standard deviations: " * join([@sprintf("%s (%.0f)", names[k], sdv[k]) for k in o], ", "))
    Z = zscore_with(X, scaler(X)); Vt, ev, evr = pca_svd(Z); cum = cumsum(evr)
    println("standardized: EVR of PC1..PC5 = " * join([@sprintf("%.3f", v) for v in evr[1:5]], ", "))
    for t in (0.8, 0.9, 0.95, 0.99)
        @printf("   components for %d%% variance: %d\n", round(Int, 100t), searchsortedfirst(cum, t))
    end
    M = fit(PCA, permutedims(Z); maxoutdim = p, pratio = 1.0)       # MultivariateStats: 열 = 관측
    k = length(principalvars(M))
    @printf("check: max |eigenvalue(SVD) - principalvars(MultivariateStats)| over %d PCs = %.2e; |<v1_svd, v1_MVS>| = %.12f\n",
            k, maximum(abs.(ev[1:k] - principalvars(M))), abs(dot(Vt[1, :], projection(M)[:, 1])))

    println("\n=== Part B: loadings ===")
    for c in 1:3
        o = sortperm(-abs.(Vt[c, :]))[1:5]
        println("PC$c: " * join([@sprintf("%s (%+.2f)", names[i], Vt[c, i]) for i in o], ", "))
    end
    T = (Z .- mean(Z; dims = 1)) * permutedims(Vt[1:2, :])
    for c in 1:2
        @printf("corr(PC%d score, band gap) = %+.3f\n", c, cor(T[:, c], y))
    end
    @printf("PC1 score: metals mean %+.2f, non-metals mean %+.2f\n", mean(T[metal, 1]), mean(T[.!metal, 1]))

    println("\n=== Part C: PCA as a regularizer (PCR) on the non-metals, grouped 5-fold CV ===")
    nm = .!metal; Xn, yn = X[nm, :], y[nm]
    Xn = Xn[:, keep_mask(Xn; thr = 0.99)]                            # 비금속에서 거름을 다시 적용
    @printf("non-metals: n = %d, p = %d after re-applying the >99%%-identical filter to this subset\n", length(yn), size(Xn, 2))
    gk = group_kfold(chemsys.(d.formula[nm]), 5)
    for k in (2, 5, 10, 20, 40, 80, 120)
        pr = cv_predict((a, b, c) -> pcr_predict(a, b, c, k), Xn, yn, gk)
        @printf("   PCR k = %3d: MAE = %.3f eV\n", k, mae(pr, yn))
    end
    pr = cv_predict((a, b, c) -> ridge_pipe(a, b, c, 31.6), Xn, yn, gk)
    @printf("   ridge (alpha = 31.6): MAE = %.3f eV\n", mae(pr, yn))

    println("\n=== Part D: metal vs non-metal from k PCs (logistic regression, grouped 5-fold) ===")
    gka = group_kfold(chemsys.(d.formula), 5); ym = Float64.(metal)
    for k in (2, 10, p)
        acc = 0
        for (tr, te) in gka
            sc = scaler(X[tr, :]); Ztr, Zte = zscore_with(X[tr, :], sc), zscore_with(X[te, :], sc)
            if k < p
                mz = mean(Ztr; dims = 1); W = permutedims(pca_svd(Ztr)[1][1:k, :])
                Ztr, Zte = (Ztr .- mz) * W, (Zte .- mz) * W
            end
            w = logreg_fit(Ztr, ym[tr]; C = 1.0)
            acc += sum((logreg_proba(w, Zte) .> 0.5) .== metal[te])
        end
        @printf("   k = %3d: accuracy = %.3f\n", k, acc / n)
    end
    @printf("time %.1f s\n", time() - t0)
end

main()
