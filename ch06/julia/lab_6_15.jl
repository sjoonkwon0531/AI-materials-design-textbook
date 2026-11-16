# Lab 6.15 (Julia) — 합성 기술자로 GWP 예측하기: 내삽은 쉽고, 외삽은 조용히 틀린다
# 파이썬 labs/66_mllca.py 대응. 자료와 "참" 법칙은 모두 합성이다:
#   ln GWP = -1.0 + 0.30 steps + 0.45 ln(MW/100) + 0.8 halogen + 0.10 n_arom + eps, eps ~ N(0, 0.35^2)
# 선형 회귀(최소제곱), RF(DecisionTree.jl, 300 트리, min_samples_leaf = 2), LightGBM.jl, 그리고
# 트리 간 퍼짐과 k-NN 거리 적용 영역(NearestNeighbors.jl)을 비교한다.
# 자료가 Julia RNG로 생성되므로 숫자는 통계적으로 일치한다. 잡음 바닥 E|eps| = 0.279는 그대로 같다.
using Printf, Random, Statistics, NearestNeighbors
include("ch6_common.jl")
include("ensemble_tools.jl")

function make(rng, n, steps_rng, mw_rng)
    steps = rand(rng, steps_rng[1]:steps_rng[2], n)
    mw = exp.(log(mw_rng[1]) .+ (log(mw_rng[2]) - log(mw_rng[1])) .* rand(rng, n))
    hal = Float64.(rand(rng, n) .< 0.25); arom = rand(rng, 0:3, n); het = 0.4 .* rand(rng, n)
    lnG = -1.0 .+ 0.30 .* steps .+ 0.45 .* log.(mw ./ 100) .+ 0.8 .* hal .+ 0.10 .* arom .+ 0.35 .* randn(rng, n)
    return hcat(steps, log.(mw ./ 100), hal, arom, het), lnG
end

function main()
    rng = MersenneTwister(3)
    Xtr, ytr = make(rng, 400, (1, 5), (30, 300))
    Xin, yin = make(rng, 400, (1, 5), (30, 300))
    Xout, yout = make(rng, 400, (6, 10), (300, 800))
    rf = rf_reg(Xtr, ytr; n_trees = 300, frac = 1.0, min_leaf = 2, seed = 0)    # sklearn RF 회귀 기본 max_features = 1.0
    bl = hcat(Xtr, ones(400)) \ ytr
    models = [("linear (ln GWP)", Z -> hcat(Z, ones(size(Z, 1))) * bl),
              ("RF", Z -> apply_forest(rf, Z)),
              ("LightGBM", Z -> lgbm_reg(Xtr, ytr, Z; num_iterations = 400, num_leaves = 15, bagging_fraction = 1.0,
                                         feature_fraction = 1.0))]
    println("=== SYNTHETIC data: train n=400 (steps 1-5, MW 30-300); test in-domain n=400; OOD n=400 (steps 6-10, MW 300-800) ===")
    @printf("%-16s %10s %11s %17s\n", "model", "MAE ln in", "MAE ln OOD", "median ratio OOD")
    pred = Dict{String,Vector{Float64}}()
    for (k, f) in models
        pin, po = f(Xin), f(Xout); pred[k] = po
        @printf("%-16s %10.3f %11.3f %17.3f\n", k, mae(pin, yin), mae(po, yout), exp(median(po .- yout)))
    end
    @printf("noise floor: E|eps| = 0.35*sqrt(2/pi) = %.3f\n", 0.35 * sqrt(2 / pi))
    @printf("true ln GWP: train max %.2f, OOD mean %.2f; RF OOD prediction max %.2f\n", maximum(ytr), mean(yout), maximum(pred["RF"]))

    spread(Z) = vec(std(tree_preds(rf, Z); dims = 1, corrected = false))
    sc = scaler(Xtr); tree = BruteTree(permutedims(zscore_with(Xtr, sc)))
    dist(Z) = [mean(d) for d in knn(tree, permutedims(zscore_with(Z, sc)), 5)[2]]
    dtr = [mean(sort(d)[2:end]) for d in knn(tree, permutedims(zscore_with(Xtr, sc)), 6)[2]]   # 자기 자신 제외
    thr = quantile(dtr, 0.95)
    s_in, s_out = spread(Xin), spread(Xout); d_in, d_out = dist(Xin), dist(Xout)
    e_in, e_out = abs.(apply_forest(rf, Xin) .- yin), abs.(pred["RF"] .- yout)
    println("=== RF tree spread vs actual error ===")
    @printf("  in-domain : mean tree sd %.3f   mean |error| %.3f\n", mean(s_in), mean(e_in))
    @printf("  OOD       : mean tree sd %.3f   mean |error| %.3f\n", mean(s_out), mean(e_out))
    @printf("=== kNN (k=5) distance in standardised descriptor space, 95th pct of training = %.3f ===\n", thr)
    @printf("  flagged as outside domain: in-domain %.3f, OOD %.3f\n", mean(d_in .> thr), mean(d_out .> thr))
    d_all = vcat(d_in, d_out); e_all = vcat(e_in, e_out)
    @printf("  mixed test set: mean |error| flagged %.3f  vs unflagged %.3f\n", mean(e_all[d_all .> thr]), mean(e_all[d_all .<= thr]))
end

main()
