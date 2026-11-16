# Lab 6.16 (Julia) — 같은 모델, 세 가지 오차
# 파이썬 labs/67_cv_leakage.py (+ 67_cv_magpie_only.py) 대응. 149개 조성 특성(Magpie 132 + Stoichiometry,
# ValenceOrbital, IonProperty)은 파이썬에서 만든 캐시 labs/67_features_clean.parquet을 CSV로 내보낸 것을 읽는다
# (matminer의 이 특성화기들은 Julia 대응이 없다; 직접 다시 만들려면 PythonCall로 matminer를 부르면 된다).
# 화학 계열은 labs/bartel_electronegativities.csv의 Pauling 값으로 Julia에서 정한다(pymatgen 판과 4,601개 모두 같다).
# LightGBM 설정은 파이썬 판과 같고, 무작위 KFold·k-평균·LightGBM 난수가 Julia 쪽이므로 숫자는 통계적으로
# 일치한다. 그룹 5겹은 동률 그룹 순서 때문에 겹 구성이 조금 다르다. 순서(무작위 < 그룹 << LOCO)는 같아야 한다.
using Printf, Random, Statistics, StatsBase, CSV, DataFrames
include("ch6_common.jl")
include("ensemble_tools.jl")

const P = (num_iterations = 400, learning_rate = 0.05, num_leaves = 31, min_data_in_leaf = 10,
           bagging_fraction = 0.8, feature_fraction = 0.5, seed = 0)

function cv_run(X, y, folds, name; clip = true)
    oof = zeros(length(y)); fm = Float64[]
    for (tr, te) in folds
        pr = lgbm_reg(X[tr, :], y[tr], X[te, :]; P...)
        oof[te] = clip ? max.(pr, 0.0) : pr; push!(fm, mae(oof[te], y[te]))
    end
    @printf("%-34s MAE = %.3f eV  (fold mean %.3f +/- %.3f)\n", name, mae(oof, y), mean(fm), std(fm; corrected = false))
    return oof, fm
end

function main()
    t0 = time()
    df = CSV.read(joinpath(LABS, "67_features_clean.csv"), DataFrame)
    feat = [c for c in names(df) if !(c in ("formula", "target", "chemsys", "reduced"))]
    X = Matrix{Float64}(df[:, feat]); y = Vector{Float64}(df.target); n = length(y)
    C = cor(zscore_with(X, scaler(X)))
    @printf("feature pairs with |r|>0.95: %d of %d\n", count(abs(C[i, j]) > 0.95 for i in 1:length(feat) for j in i+1:length(feat)),
            length(feat) * (length(feat) - 1) ÷ 2)
    en = load_en(); fam = [family(f, en) for f in df.formula]
    println("families: ", sort(collect(countmap(fam)); by = x -> -x[2]))
    @printf("baseline: predict training mean -> MAE = %.3f eV\n", mae(y, fill(mean(y), n)))
    for seed in 0:2
        cv_run(X, y, kfold(n, 5; rng = MersenneTwister(seed)), "random 5-fold (seed $seed)")
    end
    cv_run(X, y, group_kfold(String.(df.chemsys), 5), "group 5-fold (chemical system)")
    Z = zscore_with(X, scaler(X))
    for k in (5, 10)
        lab = kmeans_best(Z, k)
        folds = [(findall(!=(c), lab), findall(==(c), lab)) for c in 1:k]
        _, fm = cv_run(X, y, folds, "leave-one-cluster-out (k=$k)")
        for c in 1:k
            m = lab .== c; fc = sort(collect(countmap(fam[m])); by = x -> -x[2])
            @printf("    cluster %d: n=%4d  mean gap %.2f eV  metal %.0f%%  top family %-12s (%.0f%%)  MAE %.2f eV\n",
                    c - 1, sum(m), mean(y[m]), 100 * mean(y[m] .== 0), fc[1][1], 100 * fc[1][2] / sum(m), fm[c])
        end
    end
    # 보충: 같은 무작위 5겹(시드 0)에서 Magpie 132개만 쓰면? 자르기(>= 0) 전후
    mag = [i for (i, c) in enumerate(feat) if startswith(c, "MagpieData")]
    folds = kfold(n, 5; rng = MersenneTwister(0))
    for (nm, cols) in (("Magpie only", mag), ("all features", 1:length(feat)))
        raw = zeros(n)
        for (tr, te) in folds
            raw[te] = lgbm_reg(X[tr, cols], y[tr], X[te, cols]; P...)
        end
        @printf("%-13s (p = %3d): MAE raw = %.3f eV, clipped at 0 = %.3f eV\n", nm, length(cols), mae(raw, y), mae(max.(raw, 0), y))
    end
    @printf("wall %.0f s\n", time() - t0)
end

main()
