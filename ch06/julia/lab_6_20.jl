# Lab 6.20 (Julia) — 로지스틱, 트리, 랜덤 포레스트, Magpie 대 tau
# 파이썬 labs/68_perovskite.py의 뒷부분 대응: (a) 원시 입력 6개·공학 특성·Magpie 132개 위의 모델을 공식 분할과
# 반복 층화 5겹(5회)으로 평가, (b) 학습 곡선, (c) 음이온 계열 사이의 외삽.
# ABX3 화학식의 Magpie 특성은 두 방법 중 하나로 얻는다: (i) 파이썬에서 한 번 내보낸 labs/bartel_magpie.csv를 읽거나,
# (ii) 그 파일이 없으면 PythonCall로 matminer를 직접 부른다(아래 magpie_features).
# 로지스틱 회귀와 깊이 4 트리(cart.jl)는 결정론적이라 공식 분할의 시험 정확도가 그대로 일치해야 하고(0.810, 0.819),
# 반복 CV·RF(DecisionTree.jl)·학습 곡선의 부분표본은 Julia RNG라서 통계적으로 일치한다.
using Printf, Random, Statistics, CSV, DataFrames
include("ch6_common.jl")
include("ensemble_tools.jl")
include("cart.jl")

function magpie_features(formulas)
    path = joinpath(LABS, "bartel_magpie.csv")
    if isfile(path)
        M = CSV.read(path, DataFrame)
        @assert M.ABX3 == formulas
        return Matrix{Float64}(M[:, 2:end])
    end
    @eval using PythonCall                                     # 캐시가 없을 때만 파이썬을 부른다
    ep = Base.invokelatest(() -> pyimport("matminer.featurizers.composition").ElementProperty.from_preset("magpie"))
    Comp = Base.invokelatest(() -> pyimport("pymatgen.core").Composition)
    rows = [Base.invokelatest(() -> pyconvert(Vector{Float64}, ep.featurize(Comp(f)))) for f in formulas]
    M = permutedims(reduce(hcat, rows)); M[isnan.(M)] .= 0.0
    return M
end

"반복 층화 k겹(n_rep회)의 (학습, 시험) 목록"
repeated_strat(y; k = 5, n_rep = 5, seed = 0) = (rng = MersenneTwister(seed); reduce(vcat, [stratified_kfold(y, k; rng = rng) for _ in 1:n_rep]))

function main()
    d = CSV.read(joinpath(LABS, "bartel_TableS1.csv"), DataFrame)
    rename!(d, "rA (Ang)" => "rA", "rB (Ang)" => "rB", "rX (Ang)" => "rX")
    y = Int.(d.exp_label .> 0); tr = findall(d.is_train .== 1); te = findall(d.is_train .== -1)
    rA, rB, rX, nA = Float64.(d.rA), Float64.(d.rB), Float64.(d.rX), Float64.(d.nA)
    t = (rA .+ rX) ./ (sqrt(2) .* (rB .+ rX)); tau = rX ./ rB .- nA .* (nA .- (rA ./ rB) ./ log.(rA ./ rB)); mu = rB ./ rX
    pT = Int.(tau .< 4.18)
    raw = Matrix{Float64}(d[:, ["rA", "rB", "rX", "nA", "nB", "nX"]]); eng = hcat(raw, t, mu, tau)
    M = magpie_features(String.(d.ABX3))
    logi(C) = (a, b, c) -> (sc = scaler(a); logreg_proba(logreg_fit(zscore_with(a, sc), Float64.(b); C = C), zscore_with(c, sc)) .> 0.5)
    rf(a, b, c; n = 500, seed = 0) = apply_forest(rf_clf(a, b; n_trees = n, min_leaf = 2, seed = seed), c)
    models = [("logistic (raw 6)", logi(1.0), raw),
              ("tree depth 4 (raw 6)", (a, b, c) -> cart_predict(cart_fit(a, b; max_depth = 4), c), raw),
              ("random forest (raw 6)", rf, raw),
              ("random forest (raw 6 + t, mu, tau)", rf, eng),
              ("logistic (tau only)", (a, b, c) -> logreg_proba(logreg_fit(a, Float64.(b); C = 1e6), c) .> 0.5, reshape(tau, :, 1)),
              ("random forest (Magpie, 132)", rf, M)]
    folds = repeated_strat(y)
    println("model                                  test(split)   5x5-CV mean +/- std")
    for (k, fp, F) in models
        s_te = mean(fp(F[tr, :], y[tr], F[te, :]) .== y[te])
        s_cv = [mean(fp(F[a, :], y[a], F[b, :]) .== y[b]) for (a, b) in folds]
        @printf("%-38s %.3f        %.3f +/- %.3f\n", k, s_te, mean(s_cv), std(s_cv; corrected = false))
    end
    @printf("tau<4.18 (fixed rule) 5x5-CV-equivalent: %.3f (no fitting; single test-split value %.3f)\n", mean(pT .== y), mean(pT[te] .== y[te]))

    rng = MersenneTwister(0)
    stump(a, b, c) = cart_predict(cart_fit(a, b; max_depth = 1), c)
    sd(v) = length(v) > 1 ? std(v; corrected = false) : 0.0
    for n in (20, 40, 80, 160, 320, 460)
        acc = Dict(k => Float64[] for k in ("tau stump", "RF raw 6", "RF Magpie"))
        for rep in 1:(n < 460 ? 20 : 1)
            s = tr[randperm(rng, length(tr))[1:n]]
            length(unique(y[s])) < 2 && continue
            push!(acc["tau stump"], mean(stump(reshape(tau[s], :, 1), y[s], reshape(tau[te], :, 1)) .== y[te]))
            push!(acc["RF raw 6"], mean(rf(raw[s, :], y[s], raw[te, :]; n = 200, seed = rep) .== y[te]))
            push!(acc["RF Magpie"], mean(rf(M[s, :], y[s], M[te, :]; n = 200, seed = rep) .== y[te]))
        end
        @printf("  n=%3d  tau-stump %.3f+/-%.3f   RF raw6 %.3f+/-%.3f   RF Magpie %.3f+/-%.3f\n", n,
                mean(acc["tau stump"]), sd(acc["tau stump"]), mean(acc["RF raw 6"]), sd(acc["RF raw 6"]),
                mean(acc["RF Magpie"]), sd(acc["RF Magpie"]))
    end

    hal = in.(d.X, Ref(("F", "Cl", "Br", "I"))); oxi = .!hal
    println("extrapolation across anion families (accuracy):")
    println("  direction            tau<4.18   tau-stump(refit)   RF raw6   RF Magpie")
    for (name, a_, b_) in (("oxides -> halides", oxi, hal), ("halides -> oxides", hal, oxi))
        st = mean(stump(reshape(tau[a_], :, 1), y[a_], reshape(tau[b_], :, 1)) .== y[b_])
        r6 = mean(rf(raw[a_, :], y[a_], raw[b_, :]) .== y[b_]); rm = mean(rf(M[a_, :], y[a_], M[b_, :]) .== y[b_])
        @printf("  %-20s %.3f      %.3f              %.3f     %.3f   (n_test=%d)\n", name, mean(pT[b_] .== y[b_]), st, r6, rm, sum(b_))
    end
end

main()
