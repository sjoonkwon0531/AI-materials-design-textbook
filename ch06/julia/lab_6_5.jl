# Lab 6.5 (Julia) — Magpie 특징의 Ridge·LASSO·Elastic Net 경로와 선택의 안정성
# 파이썬 labs/63_shrinkage.py 대응. 비금속 2,151행, 준상수 특징을 뺀 123개.
# 좌표하강 LASSO/EN은 파이썬 판의 Numba 커널을 그대로 옮긴 Julia 루프이고(목적함수도 sklearn과 같다),
# 라이브러리 대조는 sklearn 대신 Lasso.jl(glmnet식 좌표하강)로 한다.
# Part A의 SVD 항등식·df(lambda)·특이값과 Part B의 목적함수 값은 그대로 일치해야 한다.
# 그룹 5겹의 겹 구성은 동률 그룹 순서 때문에 sklearn과 조금 달라 CV 곡선은 소수 둘째 자리 근처까지만 같고,
# 부트스트랩(Part D)은 Julia RNG라서 통계적으로 일치한다.
using LinearAlgebra, Printf, Random, Statistics, Lasso
include("ch6_common.jl")

"좌표하강 elastic net, 목적 (1/2n)|y - Zw|^2 + a*l1*|w|_1 + a*(1-l1)/2*|w|^2 (sklearn과 같은 척도)"
function enet_cd(Z, y, a; l1 = 1.0, w0 = zeros(size(Z, 2)), maxit = 5000, tol = 1e-9)
    n, p = size(Z); w = copy(w0); r = y - Z * w; cs = vec(sum(Z .^ 2; dims = 1)) ./ n
    it = 0
    for outer it in 1:maxit
        wmax = 0.0
        @inbounds for j in 1:p
            wj = w[j]; zj = view(Z, :, j)
            rho = dot(zj, r) / n + cs[j] * wj
            w[j] = sign(rho) * max(abs(rho) - a * l1, 0.0) / (cs[j] + a * (1 - l1))
            if w[j] != wj
                r .-= zj .* (w[j] - wj); wmax = max(wmax, abs(w[j] - wj))
            end
        end
        wmax < tol && break
    end
    return w, it
end

"절편 있는 EN (X, y를 중심화해서 풀고 절편을 되돌린다)"
function enet_fit(X, y, a; l1 = 1.0, kw...)
    mx = vec(mean(X; dims = 1)); my = mean(y)
    w, _ = enet_cd(X .- mx', y .- my, a; l1 = l1, kw...)
    return w, my - dot(mx, w)
end

function main()
    t0 = time()
    d, X, y, names = load_magpie(; nonmetal_only = true)
    keep = keep_mask(X; thr = 0.99); X = X[:, keep]; names = names[keep]
    n, p = size(X); cs = chemsys.(d.formula)
    @printf("n = %d, p = %d\n", n, p)

    # ---------- Part A: ridge = SVD shrinkage
    Z = zscore_with(X, scaler(X)); yc = y .- mean(y)
    F = svd(Z); U, s, V = F.U, F.S, F.V
    println("\n=== Part A: ridge through the SVD ===")
    for lam in (1.0, 100.0)
        f = s .^ 2 ./ (s .^ 2 .+ lam)
        w_svd = V * (f ./ map(x -> x > 1e-10 * s[1] ? x : Inf, s) .* (U' * yc))
        w_dir = ridge_fit(Z, yc, lam; intercept = false)[1]          # (Z'Z + lam I) w = Z'y
        @printf("lambda = %6.1f: max|w_SVD - w_direct| = %.2e;  df(lambda) = sum f_i = %6.2f;  ||w|| = %.3f\n",
                lam, maximum(abs.(w_svd - w_dir)), sum(f), norm(w_svd))
    end
    @printf("sigma_1^2 = %.1f, sigma_50^2 = %.2f, sigma_100^2 = %.2e; #(sigma_i^2 > 100) = %d\n",
            s[1]^2, s[50]^2, s[100]^2, count(>(100), s .^ 2))
    gkf = group_kfold(cs, 5)
    function cv_mae(fitpred)
        e = zeros(n)
        for (tr, te) in gkf
            sc = scaler(X[tr, :])
            e[te] = abs.(fitpred(zscore_with(X[tr, :], sc), y[tr], zscore_with(X[te, :], sc)) - y[te])
        end
        fold = [mean(e[te]) for (_, te) in gkf]
        return mean(e), std(fold) / sqrt(length(gkf))
    end
    lams = 10 .^ range(-3, 4; length = 15)
    rid = [cv_mae((a, b, c) -> ridge_predict(ridge_fit(a, b, l), c)) for l in lams]
    i = argmin(first.(rid))
    println("ridge grouped 5-fold CV MAE (eV) vs lambda:")
    println("  " * join([@sprintf("%.0e:%.3f", lams[k], rid[k][1]) for k in 1:2:15], "  "))
    @printf("  best lambda = %.2e, MAE = %.3f +/- %.3f\n", lams[i], rid[i][1], rid[i][2])

    # ---------- Part B: coordinate-descent LASSO vs Lasso.jl
    println("\n=== Part B: coordinate-descent LASSO (from scratch) vs Lasso.jl ===")
    amax = maximum(abs.(Z' * yc)) / n
    @printf("alpha_max = max_j |z_j^T y|/n = %.4f (all weights zero above it)\n", amax)
    obj(w, a) = 0.5 * mean((yc - Z * w) .^ 2) + a * sum(abs, w)
    for a in (0.1, 0.02)
        w_cd, it = enet_cd(Z, yc, a)
        path = fit(LassoPath, Z, yc; λ = [a], standardize = false, intercept = false, cd_tol = 1e-10, maxncoef = p)
        w_lj = Vector(coef(path)[:, end])
        @printf("alpha = %s: sweeps = %d, nonzero = %d, objective CD = %.10f, Lasso.jl = %.10f, max|w_CD - w_Lasso.jl| = %.2e\n",
                a, it, count(!iszero, w_cd), obj(w_cd, a), obj(w_lj, a), maximum(abs.(w_cd - w_lj)))
    end

    # ---------- Part C: paths and grouped CV
    alphas = 10 .^ range(log10(amax), log10(amax) - 3; length = 25)
    function path(l1)                                   # 따뜻한 시작으로 alpha를 큰 것부터 내려간다
        W = zeros(p, length(alphas)); w = zeros(p)
        for (k, a) in enumerate(alphas)
            w, _ = enet_cd(Z, yc, a; l1 = l1, w0 = w); W[:, k] = w
        end
        return W
    end
    coefs_l, coefs_e = path(1.0), path(0.5)
    las = [cv_mae((a_, b, c) -> (m = enet_fit(a_, b, a); c * m[1] .+ m[2])) for a in alphas]
    ena = [cv_mae((a_, b, c) -> (m = enet_fit(a_, b, a; l1 = 0.5); c * m[1] .+ m[2])) for a in alphas]
    nnz_l = vec(count(!iszero, coefs_l; dims = 1)); nnz_e = vec(count(!iszero, coefs_e; dims = 1))
    a1se = 0.0
    for (nm, cv, nnz) in (("LASSO", las, nnz_l), ("EN(0.5)", ena, nnz_e))
        k = argmin(first.(cv)); k1 = findfirst(c -> c[1] <= cv[k][1] + cv[k][2], cv)   # 1-SE 규칙: 가장 큰 alpha
        nm == "LASSO" && (a1se = alphas[k1])
        @printf("%-8s: best alpha = %.2e, MAE = %.3f +/- %.3f, nonzero = %d;  1-SE alpha = %.2e, MAE = %.3f, nonzero = %d\n",
                nm, alphas[k], cv[k][1], cv[k][2], nnz[k], alphas[k1], cv[k1][1], nnz[k1])
    end
    entry = [any(!iszero, coefs_l[j, :]) ? findfirst(!iszero, coefs_l[j, :]) : 999 for j in 1:p]
    println("first 8 features to enter the LASSO path (alpha at entry):")
    for j in sortperm(entry)[1:8]
        @printf("   %-28s alpha = %.3e\n", names[j], alphas[entry[j]])
    end
    w1, _ = enet_fit(Z, y, a1se)
    @printf("LASSO at the 1-SE alpha (%.2e) on all data: %d nonzero; largest |w|:\n", a1se, count(!iszero, w1))
    for j in sortperm(-abs.(w1))[1:8]
        @printf("   %-28s w = %+.3f\n", names[j], w1[j])
    end

    # ---------- Part D: stability of selection under bootstrap
    rng = MersenneTwister(0); B = 50; sel_l = zeros(p); sel_e = zeros(p)
    for _ in 1:B
        ib = rand(rng, 1:n, n); Zb = zscore_with(X[ib, :], scaler(X[ib, :]))
        sel_l .+= enet_fit(Zb, y[ib], a1se)[1] .!= 0
        sel_e .+= enet_fit(Zb, y[ib], a1se; l1 = 0.5)[1] .!= 0
    end
    sel_l ./= B; sel_e ./= B
    @printf("\n=== Part D: selection frequency over %d bootstrap resamples (alpha = %.2e) ===\n", B, a1se)
    @printf("LASSO: features selected in >=90%% of resamples: %d; in 10-90%%: %d\n", count(>=(0.9), sel_l), count(x -> 0.1 < x < 0.9, sel_l))
    @printf("EN   : features selected in >=90%% of resamples: %d; in 10-90%%: %d\n", count(>=(0.9), sel_e), count(x -> 0.1 < x < 0.9, sel_e))
    C = cor(Z)
    println("LASSO features with freq >= 0.9: " *
            join([@sprintf("%s (%.2f)", names[j], sel_l[j]) for j in sortperm(-sel_l) if sel_l[j] >= 0.9], ", "))
    for nm in ("mean Electronegativity", "maximum Electronegativity", "mean NpValence", "mean Column")
        j = findfirst(==(nm), names); j === nothing && continue
        @printf("   %-28s LASSO freq %.2f  EN freq %.2f\n", nm, sel_l[j], sel_e[j])
    end
    jA = findfirst(==("mean Electronegativity"), names); jB = findfirst(==("mean Column"), names)
    @printf("corr(mean Electronegativity, mean Column) = %.3f\n", C[jA, jB])
    jM = findfirst(==("maximum Electronegativity"), names); cc = abs.(C[jM, :]); cc[jM] = 0; jP = argmax(cc)
    @printf("maximum Electronegativity: most correlated partner = %s (corr %+.3f), its LASSO freq %.2f\n", names[jP], C[jM, jP], sel_l[jP])
    @printf("pairs of features with |corr| > 0.95: %d\n", count(abs(C[i, j]) > 0.95 for i in 1:p for j in i+1:p))
    @printf("time %.1f s\n", time() - t0)
end

main()
