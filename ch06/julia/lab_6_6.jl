# Lab 6.6 (Julia) — RBF 커널 릿지 대 선형 Ridge, 그리고 GP의 예측 분산
# 파이썬 labs/63_krr_gp.py 대응. KRR과 GP는 라이브러리 없이 선형대수로 직접 푼다:
#   KRR: a = (K + alpha I)^(-1) (y - ybar),  예측 = K(x, X) a + ybar      (sklearn KernelRidge와 같은 식)
#   GP : k = s_f^2 exp(-|x-x'|^2 / 2l^2) + s_n^2 delta, 로그 주변 우도를 Optim.jl(L-BFGS, 유한차분)로 최대화.
#        예측 표준편차는 sklearn처럼 잡음 항을 포함한다: sqrt(s_f^2 + s_n^2 - v'v).
# Part A(표현자 정리 검사)는 그대로 일치해야 하고, 그룹 5겹 숫자는 겹 구성의 동률 차이 때문에 가깝게,
# k-평균(LCO)과 GP 부분표본은 Julia RNG라서 통계적으로 일치한다.
using LinearAlgebra, Printf, Random, Statistics, StatsBase, Optim
include("ch6_common.jl")

sqdist(A, B) = max.(sum(A .^ 2; dims = 2) .+ sum(B .^ 2; dims = 2)' .- 2 .* (A * B'), 0.0)   # |a_i - b_j|^2

"KRR(RBF) 적합·예측: 학습 평균을 빼고 맞춘 뒤 되돌린다"
function krr_predict(Ztr, ytr, Zte, gamma, alpha; D = sqdist(Ztr, Ztr), Dte = sqdist(Zte, Ztr))
    m = mean(ytr); K = exp.(-gamma .* D)
    a = cholesky(Symmetric(K + alpha * I)) \ (ytr .- m)
    return exp.(-gamma .* Dte) * a .+ m
end

"GP 음의 로그 주변 우도. th = log.([s_f^2, l, s_n^2])"
function gp_nll(th, D, y)
    sf2, l, sn2 = exp.(th)
    K = sf2 .* exp.(-D ./ (2l^2)) + sn2 * I
    C = cholesky(Symmetric(K); check = false); issuccess(C) || return 1e20
    a = C \ y
    return 0.5 * dot(y, a) + sum(log, diag(C.L)) + 0.5 * length(y) * log(2pi)
end

function main()
    t0 = time()
    d, X, y, _ = load_magpie(; nonmetal_only = true)
    X = X[:, keep_mask(X; thr = 0.99)]; n, p = size(X); cs = chemsys.(d.formula)
    @printf("n = %d, p = %d\n", n, p)

    # ---------- Part A: dual = primal (linear kernel)
    Z = zscore_with(X, scaler(X)); yc = y .- mean(y); lam = 10.0
    w = ridge_fit(Z, yc, lam; intercept = false)[1]
    K = Z * Z'; a = cholesky(Symmetric(K + lam * I)) \ yc
    println("\n=== Part A: representer theorem check (linear kernel) ===")
    @printf("primal: solve a %dx%d system;  dual: solve a %dx%d system\n", p, p, n, n)
    @printf("max |Z w - K a| = %.2e;  max |w - Z^T a| = %.2e\n", maximum(abs.(Z * w - K * a)), maximum(abs.(w - Z' * a)))

    # ---------- Part B: grouped CV, ridge vs RBF-KRR
    gkf = group_kfold(cs, 5)
    clus = kmeans_best(Z, 8); lco = leave_one_group_out(clus)
    function cv_krr(folds, gamma, alpha)
        pr = zeros(n)
        for (tr, te) in folds
            sc = scaler(X[tr, :]); A_, B_ = zscore_with(X[tr, :], sc), zscore_with(X[te, :], sc)
            pr[te] = krr_predict(A_, y[tr], B_, gamma, alpha)
        end
        return pr
    end
    cv_ridge(folds) = cv_predict((a_, b, c) -> ridge_pipe(a_, b, c, 31.6), X, y, folds)
    println("\n=== Part B: grouped 5-fold CV (chemical system) ===")
    grid = Dict{Tuple{Float64,Float64},Float64}()
    for (tr, te) in gkf                                           # 겹마다 거리행렬을 한 번만 만든다
        sc = scaler(X[tr, :]); A_, B_ = zscore_with(X[tr, :], sc), zscore_with(X[te, :], sc)
        D, Dte = sqdist(A_, A_), sqdist(B_, A_)
        for g in (0.001, 0.003, 0.01, 0.03), al in (0.03, 0.1, 0.3)
            e = sum(abs.(krr_predict(A_, y[tr], B_, g, al; D = D, Dte = Dte) - y[te]))
            grid[(g, al)] = get(grid, (g, al), 0.0) + e / n
        end
    end
    println("KRR-RBF grid (gamma: MAE at alpha = 0.03 / 0.1 / 0.3):")
    for g in (0.001, 0.003, 0.01, 0.03)
        @printf("   gamma = %-6s: %s\n", g, join([@sprintf("%.3f", grid[(g, al)]) for al in (0.03, 0.1, 0.3)], " / "))
    end
    gb, ab = argmin(grid)
    sub = randperm(MersenneTwister(0), n)[1:400]; D2 = sqdist(Z[sub, :], Z[sub, :])
    med = median([D2[i, j] for i in 1:400 for j in i+1:400])
    @printf("median squared distance between standardized points = %.1f; 1/median = %.4f\n", med, 1 / med)
    pr_r = cv_ridge(gkf); pr_k = cv_krr(gkf, gb, ab)
    @printf("ridge (alpha=31.6)          : MAE = %.3f eV, RMSE = %.3f\n", mae(pr_r, y), rmse(pr_r, y))
    @printf("KRR-RBF (gamma=%s, alpha=%s): MAE = %.3f eV, RMSE = %.3f\n", gb, ab, mae(pr_k, y), rmse(pr_k, y))
    pl_r = cv_ridge(lco); pl_k = cv_krr(lco, gb, ab)
    @printf("leave-cluster-out (k=8): ridge MAE = %.3f, KRR MAE = %.3f\n", mae(pl_r, y), mae(pl_k, y))
    kmax = zeros(n)
    for (tr, te) in lco
        sc = scaler(X[tr, :])
        kmax[te] = exp.(-gb .* vec(minimum(sqdist(zscore_with(X[te, :], sc), zscore_with(X[tr, :], sc)); dims = 2)))
    end
    @printf("LCO: median over test points of max_j k(x, x_j) = %.3f\n", median(kmax))
    far = kmax .< 0.2
    @printf("LCO test points with max kernel < 0.2: %d; their KRR MAE = %.3f, ridge MAE = %.3f, |KRR pred - train mean| median = %.3f\n",
            sum(far), mae(pl_k[far], y[far]), mae(pl_r[far], y[far]), median(abs.(pl_k[far] .- mean(y))))

    # ---------- Part C: GP preview
    println("\n=== Part C: Gaussian process = Bayesian KRR with a predictive variance ===")
    tr, te = gkf[1]
    tr = tr[randperm(MersenneTwister(1), length(tr))[1:1000]]
    sc = scaler(X[tr, :]); Ztr, Zte = zscore_with(X[tr, :], sc), zscore_with(X[te, :], sc); m0 = mean(y[tr])
    D = sqdist(Ztr, Ztr); ytr = y[tr] .- m0
    res = optimize(th -> gp_nll(th, D, ytr), log.([1.0, 10.0, 0.1]), LBFGS())
    sf2, ell, sn2 = exp.(Optim.minimizer(res))
    @printf("train n = %d (subsample of fold-0 training set), test n = %d\n", length(tr), length(te))
    @printf("learned kernel: %.2f**2 * RBF(length_scale=%.2f) + WhiteKernel(noise_level=%.3g)\n", sqrt(sf2), ell, sn2)
    C = cholesky(Symmetric(sf2 .* exp.(-D ./ (2ell^2)) + sn2 * I))
    Ks = sf2 .* exp.(-sqdist(Zte, Ztr) ./ (2ell^2))
    mu = Ks * (C \ ytr) .+ m0
    v = C.L \ permutedims(Ks)
    sd = sqrt.(max.(sf2 + sn2 .- vec(sum(v .^ 2; dims = 1)), 0.0))
    kr = krr_predict(Ztr, y[tr], Zte, 1 / (2ell^2), sn2 / sf2)
    @printf("max |GP mean - KRR(gamma=1/2l^2, alpha=s_n^2/s_f^2)| = %.2e\n", maximum(abs.(kr - mu)))
    err = abs.(mu - y[te])
    @printf("GP test MAE = %.3f eV; mean predictive std = %.3f eV\n", mean(err), mean(sd))
    for (z, nom) in ((1.0, 0.683), (1.96, 0.95))
        @printf("   coverage of mu +/- %s*std: %.3f (nominal %s)\n", z, mean(err .<= z .* sd), nom)
    end
    @printf("Spearman rank corr(std, |error|) = %.3f\n", corspearman(sd, err))
    q = quantile(sd, [0, 0.25, 0.5, 0.75, 1.0])
    for (lo, hi) in zip(q[1:end-1], q[2:end])
        msk = (sd .>= lo) .& (sd .<= hi)
        @printf("   std in [%.2f, %.2f]: n = %3d, MAE = %.3f\n", lo, hi, sum(msk), mean(err[msk]))
    end
    @printf("time %.1f s\n", time() - t0)
end

main()
