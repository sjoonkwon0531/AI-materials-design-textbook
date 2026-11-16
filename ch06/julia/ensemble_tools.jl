# ensemble_tools.jl — 트리·앙상블 모델의 얇은 감싸개 (DecisionTree.jl, XGBoost.jl 2.x, LightGBM.jl)
# sklearn 이름과 대응: max_features(비율) -> n_subfeatures(정수), bootstrap -> partial_sampling = 1.0,
# min_samples_leaf는 DecisionTree.jl 회귀의 기본값이 5이므로 sklearn 기본값 1을 명시한다.
# XGBoost.jl의 xgboost((X, y); ...)와 LightGBM.jl의 LGBMRegression/LGBMClassification 키워드는
# 각 라이브러리의 원래 매개변수 이름(eta, colsample_bytree / bagging_fraction, feature_fraction ...)을 쓴다.
using DecisionTree, XGBoost, LightGBM, Random

mf(p, frac) = max(1, floor(Int, frac * p))          # sklearn max_features = 비율 -> 정수

"랜덤 포레스트(회귀). frac = 분기마다 고르는 특징 비율"
rf_reg(X, y; n_trees = 100, frac = 1 / 3, min_leaf = 1, seed = 0) =
    build_forest(y, X, mf(size(X, 2), frac), n_trees, 1.0, -1, min_leaf, 2, 0.0; rng = MersenneTwister(seed))

"랜덤 포레스트(분류). sklearn 기본 max_features = sqrt(p)"
rf_clf(X, y; n_trees = 300, nsub = floor(Int, sqrt(size(X, 2))), min_leaf = 1, seed = 0) =
    build_forest(y, X, nsub, n_trees, 1.0, -1, min_leaf, 2, 0.0; rng = MersenneTwister(seed))

"트리 하나하나의 예측: (트리 수) x (관측 수)"
tree_preds(forest, X) = permutedims(reduce(hcat, [apply_tree(t, X) for t in forest.trees]))

"단일 결정 트리(회귀·분류 공통): max_depth = -1이면 끝까지"
tree_fit(X, y; max_depth = -1, min_leaf = 1, seed = 0) =
    build_tree(y, X, 0, max_depth, min_leaf, 2, 0.0; rng = MersenneTwister(seed))

"XGBoost 회귀/분류. objective = \"reg:squarederror\" 또는 \"binary:logistic\""
function xgb_fit_predict(Xtr, ytr, Xte; num_round = 400, eta = 0.05, max_depth = 6, subsample = 0.8,
                         colsample_bytree = 0.5, min_child_weight = 1, lambda = 1.0, seed = 0,
                         objective = "reg:squarederror")
    bst = xgboost((Xtr, ytr); num_round = num_round, eta = eta, max_depth = max_depth, subsample = subsample,
                  colsample_bytree = colsample_bytree, min_child_weight = min_child_weight, lambda = lambda,
                  objective = objective, nthread = 1, seed = seed, watchlist = Dict())
    return XGBoost.predict(bst, Xte)
end

"LightGBM 회귀 모델 객체. sklearn 이름 -> LightGBM 이름: n_estimators=num_iterations, subsample=bagging_fraction,
subsample_freq=bagging_freq, colsample_bytree=feature_fraction, min_child_samples=min_data_in_leaf"
function lgbm_reg_model(Xtr, ytr; num_iterations = 400, learning_rate = 0.05, num_leaves = 31,
                        min_data_in_leaf = 20, bagging_fraction = 0.8, feature_fraction = 0.5, seed = 0,
                        objective = "regression", alpha = 0.9)
    est = LGBMRegression(; objective = objective, alpha = alpha, num_iterations = num_iterations,
                         learning_rate = learning_rate, num_leaves = num_leaves, min_data_in_leaf = min_data_in_leaf,
                         bagging_fraction = bagging_fraction, bagging_freq = 1, feature_fraction = feature_fraction,
                         num_threads = 1, seed = seed, verbosity = -1)
    LightGBM.fit!(est, Xtr, ytr; verbosity = -1)
    return est
end
lgbm_out(est, X) = vec(LightGBM.predict(est, X; verbosity = -1))
lgbm_reg(Xtr, ytr, Xte; kw...) = lgbm_out(lgbm_reg_model(Xtr, ytr; kw...), Xte)

"LightGBM 이진 분류 모델 객체 (lgbm_out으로 양성 확률을 얻는다)"
function lgbm_clf_model(Xtr, ytr; num_iterations = 400, learning_rate = 0.05, num_leaves = 31,
                        min_data_in_leaf = 20, bagging_fraction = 0.8, feature_fraction = 0.5, seed = 0)
    est = LGBMClassification(; objective = "binary", num_class = 1, num_iterations = num_iterations,
                             learning_rate = learning_rate, num_leaves = num_leaves, min_data_in_leaf = min_data_in_leaf,
                             bagging_fraction = bagging_fraction, bagging_freq = 1, feature_fraction = feature_fraction,
                             num_threads = 1, seed = seed, verbosity = -1)
    LightGBM.fit!(est, Xtr, Float64.(ytr); verbosity = -1)
    return est
end
lgbm_clf(Xtr, ytr, Xte; kw...) = lgbm_out(lgbm_clf_model(Xtr, ytr; kw...), Xte)
