# Lab 6.11 (Julia) — 특성 중요도는 무엇을 재는가
# 파이썬 labs/65_importance.py 대응. RF는 DecisionTree.jl(build_forest), 불순도 중요도(MDI)는
# DecisionTree.impurity_importance, 순열 중요도는 시험 MAE 증가량을 정의대로 직접 계산한다.
# 분할·잡음 탐침·RF가 모두 Julia RNG를 쓰므로 순위와 값은 통계적으로만 일치한다. 볼 것: 잡음 탐침의 MDI
# 순위가 중간쯤인데 순열 중요도는 0 근처(맨 아래)라는 것, 거의 복제본을 넣으면 두 중요도가 나뉜다는 것.
using Printf, Random, Statistics, StatsBase
include("ch6_common.jl")
include("ensemble_tools.jl")

"순열 중요도: 열 j를 섞었을 때 시험 MAE 증가량의 평균(n_repeats회)"
function permutation_importance(predict, X, y; n_repeats = 5, rng = MersenneTwister(0))
    base = mae(predict(X), y); imp = zeros(size(X, 2))
    for j in 1:size(X, 2), _ in 1:n_repeats
        Xp = copy(X); Xp[:, j] = shuffle(rng, Xp[:, j])
        imp[j] += (mae(predict(Xp), y) - base) / n_repeats
    end
    return imp
end

function fit_rank(Xtr, ytr, Xte, yte, tag)
    rf = rf_reg(Xtr, ytr; n_trees = 300, frac = 0.33, seed = 0)
    mdi = impurity_importance(rf; normalize = true)
    perm = permutation_importance(Z -> apply_forest(rf, Z), Xte, yte)
    @printf("--- %s: test MAE = %.3f eV ---\n", tag, mae(apply_forest(rf, Xte), yte))
    return mdi, perm
end

function main()
    _, X, y, names = load_magpie()
    rng = MersenneTwister(0)
    X = hcat(X, randn(rng, length(y))); names = vcat(names, "PROBE noise")
    idx = randperm(MersenneTwister(0), length(y)); nte = ceil(Int, 0.2 * length(y))
    te, tr = idx[1:nte], idx[nte+1:end]                       # train_test_split(test_size = 0.2)
    Xtr, Xte, ytr, yte = X[tr, :], X[te, :], y[tr], y[te]

    mdi, perm = fit_rank(Xtr, ytr, Xte, yte, "RF with noise probe")
    println("top 8 by MDI (impurity)            | top 8 by permutation (test MAE increase, eV)")
    a, b = sortperm(-mdi)[1:8], sortperm(-perm)[1:8]
    for (i, j) in zip(a, b)
        @printf("  %-30s %.3f   | %-30s %.3f\n", names[i], mdi[i], names[j], perm[j])
    end
    k = length(names)
    r_mdi = ordinalrank(-mdi)[k]; r_perm = ordinalrank(-perm)[k]
    @printf("noise probe: MDI = %.4f (rank %d/%d), permutation = %+.4f eV (rank %d)\n", mdi[k], r_mdi, k, perm[k], r_perm)
    @printf("  number of real Magpie features with MDI below the pure-noise probe: %d\n", count(<(mdi[k]), mdi))
    top = argmax(perm)
    c = [abs(cor(Xtr[:, j], Xtr[:, top])) for j in 1:k-1]
    @printf("features with |r|>0.9 with '%s': %d\n", names[top], count(>(0.9), c) - 1)

    s = std(Xtr[:, top])
    X2tr = hcat(Xtr, Xtr[:, top] .+ 0.01 * s .* randn(rng, length(ytr)))
    X2te = hcat(Xte, Xte[:, top] .+ 0.01 * s .* randn(rng, length(yte)))
    mdi2, perm2 = fit_rank(X2tr, ytr, X2te, yte, "RF with near-duplicate of the top feature")
    @printf("  '%s': MDI %.3f -> %.3f (+ duplicate %.3f); permutation %.3f -> %.3f (duplicate %.3f) eV\n",
            names[top], mdi[top], mdi2[top], mdi2[end], perm[top], perm2[top], perm2[end])
end

main()
