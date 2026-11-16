# Lab 6.13 (Julia) — expt_gap에서의 단일 트리, RF, XGBoost, LightGBM, 스태킹
# 파이썬 labs/65_ensemble.py, 65_lc.py, 65_ridge_outlier.py 대응.
#  (A) 회귀 5겹: 평균 예측, ridge(표준화, alpha = 10), 단일 트리, RF, XGBoost, LightGBM, OOF 스태킹(메타 모델:
#      음이 아닌 선형 회귀 = 중심화한 자료의 NNLS + 절편), 누수가 있는 순진한 스태킹.
#  (B) 금속/비금속 분류 5겹: 로지스틱, 단일 트리, AdaBoost(그루터기; DecisionTree.jl의 SAMME는 학습률 1로
#      고정이라 sklearn의 0.5와 다르다), RF, XGBoost, LightGBM. 깊이 3 AdaBoost는 DecisionTree.jl에 없어 뺐다.
#  (C) 학습 곡선, (D) ridge가 한 겹에서 폭발하는 이유.
# 분할과 모델 난수가 Julia RNG이므로 숫자는 통계적으로 일치한다. (D)의 Fe9Co7 폭발은 분할과 무관하게 재현된다
# ('minimum GSmagmom'이 0이 아닌 유일한 큰 값이 시험 겹에 들어가면 학습 겹의 표준편차가 거의 0이 되기 때문).
# 같은 모델을 MLJ.jl의 evaluate로 평가하는 패턴은 lab_6_13_mlj.jl에 따로 두었다.
using Printf, Random, Statistics, LinearAlgebra, NonNegLeastSquares
include("ch6_common.jl")
include("ensemble_tools.jl")

reg_models() = [
    ("ridge", (a, b, c) -> ridge_pipe(a, b, c, 10.0)),
    ("tree", (a, b, c) -> apply_tree(tree_fit(a, b; min_leaf = 3), c)),
    ("RF", (a, b, c) -> apply_forest(rf_reg(a, b; n_trees = 200, frac = 0.33), c)),
    ("XGBoost", (a, b, c) -> xgb_fit_predict(a, b, c; num_round = 600, min_child_weight = 2)),
    ("LightGBM", (a, b, c) -> lgbm_reg(a, b, c; num_iterations = 600, min_data_in_leaf = 10)),
]
const BASE = ["ridge", "RF", "XGBoost", "LightGBM"]

"sklearn LinearRegression(positive=True): 중심화 후 NNLS, 절편 = ybar - zbar.w"
function nnls_meta(Z, y)
    mz = vec(mean(Z; dims = 1)); my = mean(y)
    w = vec(nonneg_lsq(Z .- mz', y .- my; alg = :nnls))
    return vcat(my - dot(mz, w), w)
end
meta_predict(wb, Z) = Z * wb[2:end] .+ wb[1]

function part_a(X, y)
    names_ = vcat("dummy", first.(reg_models()), "stack (OOF)", "stack (naive)")
    oof = Dict(k => zeros(length(y)) for k in names_); maes = Dict(k => Float64[] for k in names_)
    w_oof, w_naive = Vector{Float64}[], Vector{Float64}[]
    t0 = time()
    for (fold, (tr, te)) in enumerate(kfold(length(y), 5; rng = MersenneTwister(0)))
        Xtr, ytr, Xte = X[tr, :], y[tr], X[te, :]
        oof["dummy"][te] .= mean(ytr)
        full_te, full_tr = Dict{String,Vector{Float64}}(), Dict{String,Vector{Float64}}()
        for (k, fp) in reg_models()
            pr = fp(Xtr, ytr, vcat(Xte, Xtr))                 # 한 번 맞추고 시험·학습을 함께 예측
            full_te[k] = pr[1:length(te)]; full_tr[k] = pr[length(te)+1:end]; oof[k][te] = full_te[k]
        end
        Zin = zeros(length(tr), length(BASE))                 # 안쪽 3겹 OOF 예측 (메타 모델의 학습 자료)
        for (itr, ite) in kfold(length(tr), 3; rng = MersenneTwister(1))
            ms = Dict(reg_models())
            for (j, k) in enumerate(BASE)
                Zin[ite, j] = ms[k](Xtr[itr, :], ytr[itr], Xtr[ite, :])
            end
        end
        Zte = reduce(hcat, [full_te[k] for k in BASE])
        wo = nnls_meta(Zin, ytr); oof["stack (OOF)"][te] = meta_predict(wo, Zte); push!(w_oof, wo)
        wn = nnls_meta(reduce(hcat, [full_tr[k] for k in BASE]), ytr)      # 표본 내 예측: 누수
        oof["stack (naive)"][te] = meta_predict(wn, Zte); push!(w_naive, wn)
        for k in names_
            push!(maes[k], mae(oof[k][te], y[te]))
        end
        @printf("  fold %d done (%.0f s)\n", fold - 1, time() - t0)
    end
    println("=== Part A: regression, 5-fold CV (shuffled KFold) ===")
    @printf("%-15s %9s %7s %6s  %9s\n", "model", "MAE mean", "+/- sd", "RMSE", "MAE gap>0")
    nz = y .> 0
    for k in names_
        @printf("%-15s %9.3f %7.3f %6.3f  %9.3f\n", k, mean(maes[k]), std(maes[k]), rmse(oof[k], y), mae(oof[k][nz], y[nz]))
    end
    fmt(v) = "[" * join([@sprintf("%.3f", x) for x in v], " ") * "]"
    println("meta weights [intercept, ", join(BASE, ", "), "] (mean over folds)")
    println("  OOF  : ", fmt(mean(w_oof))); println("  naive: ", fmt(mean(w_naive)))
    Rm = reduce(hcat, [oof[k] .- y for k in BASE])
    println("correlation of out-of-fold residuals (ridge, RF, XGB, LGBM):")
    display(round.(cor(Rm); digits = 3)); println()
    return oof
end

function part_b(X, y)
    yc = Int.(y .> 0)
    cls = [
        ("logistic", (a, b, c) -> (sc = scaler(a); logreg_proba(logreg_fit(zscore_with(a, sc), Float64.(b); C = 1.0), zscore_with(c, sc)))),
        ("tree", (a, b, c) -> apply_tree_proba(tree_fit(a, b; min_leaf = 3), c, [0, 1])[:, 2]),
        ("AdaBoost", (a, b, c) -> (m = build_adaboost_stumps(b, a, 300; rng = MersenneTwister(0));
                                   apply_adaboost_stumps_proba(m[1], m[2], c, [0, 1])[:, 2])),
        ("RF", (a, b, c) -> apply_forest_proba(rf_clf(a, b; n_trees = 300), c, [0, 1])[:, 2]),
        ("XGBoost", (a, b, c) -> xgb_fit_predict(a, Float64.(b), c; num_round = 600, objective = "binary:logistic")),
        ("LightGBM", (a, b, c) -> lgbm_clf(a, b, c; num_iterations = 600)),
    ]
    println("=== Part B: metal (gap=0) vs non-metal, 5-fold stratified CV ===")
    @printf("%-12s %9s %8s\n", "model", "accuracy", "ROC-AUC")
    folds = stratified_kfold(yc, 5; rng = MersenneTwister(0))
    for (k, fp) in cls
        acc, auc = Float64[], Float64[]
        for (tr, te) in folds
            pr = fp(X[tr, :], yc[tr], X[te, :])
            push!(acc, mean((pr .> 0.5) .== yc[te])); push!(auc, roc_auc(yc[te], pr))
        end
        @printf("%-12s %9.3f %8.3f   (acc sd %.3f)\n", k, mean(acc), mean(auc), std(acc))
    end
end

function part_c(X, y)
    mk = [("ridge", (a, b, c) -> ridge_pipe(a, b, c, 10.0)),
          ("tree", (a, b, c) -> apply_tree(tree_fit(a, b; min_leaf = 3), c)),
          ("RF", (a, b, c) -> apply_forest(rf_reg(a, b; n_trees = 150, frac = 0.33), c)),
          ("LightGBM", (a, b, c) -> lgbm_reg(a, b, c; num_iterations = 600, min_data_in_leaf = 10))]
    sizes = [250, 500, 1000, 2000, 3680]
    te_m = Dict(k => zeros(5, 5) for (k, _) in mk); tr_m = Dict(k => zeros(5, 5) for (k, _) in mk)
    for (f, (tr, te)) in enumerate(kfold(length(y), 5; rng = MersenneTwister(0)))
        order = tr[randperm(MersenneTwister(f), length(tr))]
        for (j, n) in enumerate(sizes)
            sub = order[1:min(n, length(order))]
            for (k, fp) in mk
                pr = fp(X[sub, :], y[sub], vcat(X[te, :], X[sub, :]))
                te_m[k][f, j] = mae(pr[1:length(te)], y[te]); tr_m[k][f, j] = mae(pr[length(te)+1:end], y[sub])
            end
        end
    end
    println("=== Part C: learning curves: test MAE (eV), mean +/- sd over 5 folds; train MAE in brackets ===")
    println("n_train  " * join([lpad(k, 20) for (k, _) in mk], "  "))
    for (j, n) in enumerate(sizes)
        println(lpad(n, 7), "  ", join([@sprintf("%6.3f+/-%.3f [%.3f]", mean(te_m[k][:, j]), std(te_m[k][:, j]), mean(tr_m[k][:, j])) for (k, _) in mk], "  "))
    end
    for k in ("RF", "LightGBM")
        b = hcat(ones(4), log.(sizes[2:end])) \ log.(vec(mean(te_m[k]; dims = 1))[2:end])
        @printf("%s: log-log slope of test MAE vs n (n>=500) = %.3f\n", k, b[2])
    end
end

function part_d(X, y, d, names)
    println("=== Part D: why ridge explodes in one fold ===")
    oof = zeros(length(y)); zmax = zeros(length(y)); zfeat = fill("", length(y))
    for (f, (tr, te)) in enumerate(kfold(length(y), 5; rng = MersenneTwister(0)))
        sc = scaler(X[tr, :]); m = ridge_fit(zscore_with(X[tr, :], sc), y[tr], 10.0)
        Zte = abs.(zscore_with(X[te, :], sc)); oof[te] = ridge_predict(m, zscore_with(X[te, :], sc))
        zmax[te] = vec(maximum(Zte; dims = 2)); zfeat[te] = names[[argmax(r) for r in eachrow(Zte)]]
        @printf("fold %d: ridge MAE = %.3f  max |pred| = %.1f eV\n", f - 1, mae(oof[te], y[te]), maximum(abs.(oof[te])))
    end
    for i in sortperm(-abs.(oof .- y))[1:3]
        @printf("  %-22s true %.2f  ridge %9.1f eV  max|z| = %.0f (%s)\n", d.formula[i], y[i], oof[i], zmax[i], zfeat[i])
    end
    keep = abs.(oof .- y) .< 100
    @printf("ridge MAE without the %d compound(s) with |error| > 100 eV: %.3f\n", sum(.!keep), mae(oof[keep], y[keep]))
    @printf("ridge MAE with predictions clipped to [0, %.1f] eV: %.3f\n", maximum(y), mae(clamp.(oof, 0, maximum(y)), y))
end

function main()
    d, X, y, names = load_magpie()
    @printf("data: n=%d, p=%d Magpie features, metals (gap=0) = %.3f\n", length(y), size(X, 2), mean(y .== 0))
    part_a(X, y); part_b(X, y); part_c(X, y); part_d(X, y, d, names)
end

main()
