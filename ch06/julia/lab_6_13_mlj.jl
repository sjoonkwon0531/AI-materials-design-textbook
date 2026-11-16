# Lab 6.13 보충 (Julia) — 같은 5겹 위에서 MLJ.jl의 evaluate로 RF·XGBoost·LightGBM을 평가하는 패턴
# MLJ는 모델 인터페이스를 통일하므로 하이퍼파라미터 이름만 각 패키지의 것을 쓰면 된다.
# 겹은 lab_6_13.jl과 같은 kfold(시드 0)를 (train, test) 쌍의 벡터로 넘긴다.
using MLJ, Random, Printf
include("ch6_common.jl")

RFR = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
XGBR = @load XGBoostRegressor pkg = XGBoost verbosity = 0
LGBR = @load LGBMRegressor pkg = LightGBM verbosity = 0

function main()
    _, X, y, _ = load_magpie(); p = size(X, 2)
    Xt = MLJ.table(X)
    folds = kfold(length(y), 5; rng = MersenneTwister(0))
    models = [
        ("RF", RFR(; n_trees = 200, n_subfeatures = floor(Int, 0.33p), sampling_fraction = 1.0, min_samples_leaf = 1,
                   rng = MersenneTwister(0))),
        ("XGBoost", XGBR(; num_round = 600, eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.5,
                         min_child_weight = 2, lambda = 1.0, nthread = 1)),
        ("LightGBM", LGBR(; num_iterations = 600, learning_rate = 0.05, num_leaves = 31, bagging_fraction = 0.8,
                          bagging_freq = 1, feature_fraction = 0.5, min_data_in_leaf = 10, num_threads = 1)),
    ]
    for (name, model) in models
        e = evaluate(model, Xt, y; resampling = folds, measure = l1, verbosity = 0)
        @printf("%-9s MAE = %.3f  (per fold: %s)\n", name, e.measurement[1], join([@sprintf("%.3f", v) for v in e.per_fold[1]], ", "))
    end
end

main()
