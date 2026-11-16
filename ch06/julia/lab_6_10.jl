# Lab 6.10 (Julia) — 평균의 수학을 재다: rho, sigma^2, 그리고 bias-variance
# 파이썬 labs/65_bv.py 대응. 랜덤 포레스트·배깅·단일 트리는 DecisionTree.jl(트리별 예측을 꺼내
# Var = rho s^2 + (1 - rho) s^2 / B를 검사), XGBoost.jl, LightGBM.jl을 쓴다.
# 시험 400개·풀·학습 집합 추출과 모든 모델의 난수가 Julia RNG이므로 숫자는 통계적으로 일치한다.
# 그대로 성립해야 하는 것: Part A의 B = 100 행에서 경험 분산 = 예측식(정의상 같은 양), Part B의 항등식
# bias^2 + var - MSE = 0 (반올림 오차 수준).
using Printf, Random, Statistics
include("ch6_common.jl")
include("ensemble_tools.jl")

function main()
    t0 = time()
    _, X, y, _ = load_magpie(); n, p = size(X)
    perm = randperm(MersenneTwister(0), n)
    n_test, n_train, R, B = 400, 1500, 16, 100
    te, pool = perm[1:n_test], perm[n_test+1:end]
    Xte, yte = X[te, :], y[te]
    draw(r) = pool[randperm(MersenneTwister(100 + r), length(pool))[1:n_train]]   # 비복원 추출

    @printf("=== Part A: averaged trees, n_train=%d, R=%d replicate training sets, B=%d, n_test=%d ===\n", n_train, R, B, n_test)
    Bgrid = (1, 2, 5, 10, 20, 50, 100)
    for (frac, name) in ((1.0, "bagging (m=p)"), (0.33, "RF m=p/3"), (0.1, "RF m=p/10"))
        P = zeros(R, B, n_test)
        for r in 1:R
            tr = draw(r)
            P[r, :, :] = tree_preds(rf_reg(X[tr, :], y[tr]; n_trees = B, frac = frac, seed = r), Xte)
        end
        mu = dropdims(mean(P; dims = (1, 2)); dims = (1, 2))                  # 데이터·트리 난수에 대한 기댓값
        D = P .- reshape(mu, 1, 1, :)
        sigma2 = dropdims(mean(D .^ 2; dims = (1, 2)); dims = (1, 2))         # 시험점마다 단일 트리 분산
        S = dropdims(sum(D; dims = 2); dims = 2)                              # R x n_test
        cov = vec(mean((S .^ 2 .- dropdims(sum(D .^ 2; dims = 2); dims = 2)) ./ (B * (B - 1)); dims = 1))
        s2, c = mean(sigma2), mean(cov); rho = c / s2
        bias2 = mean((yte .- mu) .^ 2)
        maeB = mean(abs.(dropdims(mean(P; dims = 2); dims = 2) .- yte'))
        @printf("%-14s mean sigma^2 = %.3f  rho(sigma^2-weighted) = %.3f  bias^2+noise = %.3f  test MAE (B=%d) = %.3f eV\n",
                name, s2, rho, bias2, B, maeB)
        for Bk in Bgrid
            m = dropdims(mean(P[:, 1:Bk, :]; dims = 2); dims = 2)             # 앞의 Bk개 트리로 된 숲
            emp = mean(mean((m .- mu') .^ 2; dims = 1))
            @printf("    B=%3d: Var(empirical) = %.4f   rho*s2+(1-rho)*s2/B = %.4f\n", Bk, emp, rho * s2 + (1 - rho) * s2 / Bk)
        end
    end
    @printf("Part A time %.1f s\n", time() - t0)

    t1 = time()
    @printf("=== Part B: bias^2(+noise) / variance over R=%d training sets of n=%d ===\n", R, n_train)
    models = [
        ("tree (full depth)", (a, b, c, s) -> apply_tree(tree_fit(a, b; seed = s), c)),
        ("tree (depth 4)", (a, b, c, s) -> apply_tree(tree_fit(a, b; max_depth = 4, seed = s), c)),
        ("bagging B=100", (a, b, c, s) -> apply_forest(rf_reg(a, b; n_trees = 100, frac = 1.0, seed = s), c)),
        ("RF B=100, m=p/3", (a, b, c, s) -> apply_forest(rf_reg(a, b; n_trees = 100, frac = 0.33, seed = s), c)),
        ("XGBoost", (a, b, c, s) -> xgb_fit_predict(a, b, c; seed = s)),
        ("LightGBM", (a, b, c, s) -> lgbm_reg(a, b, c; seed = s)),
    ]
    @printf("%-20s %9s %12s %9s %7s\n", "model", "test MSE", "bias2+noise", "variance", "MAE")
    for (name, fp) in models
        P = zeros(R, n_test)
        for r in 1:R
            tr = draw(r); P[r, :] = fp(X[tr, :], y[tr], Xte, r)
        end
        Fbar = vec(mean(P; dims = 1))
        mse = mean((P .- yte') .^ 2); b2 = mean((Fbar .- yte) .^ 2); var_ = mean((P .- Fbar') .^ 2)
        @printf("%-20s %9.3f %12.3f %9.3f %7.3f   (check: b2+var-mse = %+.1e)\n", name, mse, b2, var_, mean(abs.(P .- yte')), b2 + var_ - mse)
    end
    @printf("Part B time %.1f s; total %.1f s\n", time() - t1, time() - t0)
end

main()
