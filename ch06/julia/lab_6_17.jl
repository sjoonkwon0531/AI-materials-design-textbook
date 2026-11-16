# Lab 6.17 (Julia) — 정직한 홀드아웃 위의 파이프라인
# 파이썬 labs/67_pipeline.py 대응: 화학계 단위 20% 홀드아웃을 끝까지 남겨 두고, 학습 부분 안에서만
# (a) 그룹 3겹 하이퍼파라미터 탐색, (b) 직접 회귀 대 2단계(분류기 + 비금속 회귀기; 하드·소프트 게이트),
# (c) 계열별 오차, (d) 순열 중요도, (e) 분류 확률의 등위 보정(sklearn CalibratedClassifierCV(method="isotonic")와
# 같은 절차: 겹마다 분류기를 맞추고 떼어 둔 부분에서 등위 회귀를 맞춘 뒤 확률을 평균), (f) 분위 LightGBM 구간과
# RF 퍼짐의 분할 등각(conformal) 재보정. 등위 회귀(PAVA)는 직접 구현했다.
# 홀드아웃의 그룹 추출과 모든 모델 난수가 Julia RNG이므로 숫자는 통계적으로 일치한다.
using Printf, Random, Statistics, StatsBase, CSV, DataFrames, Distributions
include("ch6_common.jl")
include("ensemble_tools.jl")

"등위 회귀(PAVA, 증가). 같은 x는 먼저 평균한다. 반환: 단조 계단의 (x 마디, 값) -> 선형 보간 + 양끝 자르기"
function isotonic_fit(x, y)
    o = sortperm(x); xs, ys = x[o], y[o]
    ux = Float64[]; uy = Float64[]; uw = Float64[]
    for (a, b) in zip(xs, ys)                                  # 같은 x 묶기
        if !isempty(ux) && a == ux[end]
            uy[end] = (uy[end] * uw[end] + b) / (uw[end] + 1); uw[end] += 1
        else
            push!(ux, a); push!(uy, b); push!(uw, 1.0)
        end
    end
    v = Float64[]; w = Float64[]; len = Int[]                  # PAVA 블록
    for (b, c) in zip(uy, uw)
        push!(v, b); push!(w, c); push!(len, 1)
        while length(v) > 1 && v[end-1] > v[end]
            nw = w[end-1] + w[end]; v[end-1] = (v[end-1] * w[end-1] + v[end] * w[end]) / nw
            w[end-1] = nw; len[end-1] += len[end]; pop!(v); pop!(w); pop!(len)
        end
    end
    fit = reduce(vcat, [fill(v[k], len[k]) for k in eachindex(v)])
    return ux, fit
end
function isotonic_predict((ux, fy), x)
    map(x) do t
        t <= ux[1] && return fy[1]; t >= ux[end] && return fy[end]
        k = searchsortedlast(ux, t); s = (t - ux[k]) / (ux[k+1] - ux[k])
        return fy[k] + s * (fy[k+1] - fy[k])
    end
end

function ece(p, t; nb = 10)
    b = min.(floor.(Int, p .* nb), nb - 1)
    return sum(abs(mean(p[b .== k]) - mean(t[b .== k])) * mean(b .== k) for k in 0:nb-1 if any(b .== k))
end

function main()
    t0 = time()
    df = CSV.read(joinpath(LABS, "67_features_clean.csv"), DataFrame)
    feat = [c for c in names(df) if !(c in ("formula", "target", "chemsys", "reduced"))]
    X = Matrix{Float64}(df[:, feat]); y = Vector{Float64}(df.target); g = String.(df.chemsys)
    en = load_en(); fam = [family(f, en) for f in df.formula]
    tr, te = group_shuffle_split(g; test_size = 0.2, rng = MersenneTwister(42))
    @printf("train %d / test %d (chemical systems disjoint: %s)\n", length(tr), length(te), isempty(intersect(Set(g[tr]), Set(g[te]))))
    base = (bagging_fraction = 0.8, feature_fraction = 0.5, seed = 0, num_iterations = 400)

    # (a) 학습 부분 안에서만 그룹 3겹 탐색
    best = (Inf, nothing)
    for nl in (15, 63), lr in (0.03, 0.1), mcs in (5, 20)
        hp = (num_leaves = nl, learning_rate = lr, min_data_in_leaf = mcs)
        sc = [mae(y[tr][b], max.(lgbm_reg(X[tr[a], :], y[tr[a]], X[tr[b], :]; hp..., base...), 0)) for (a, b) in group_kfold(g[tr], 3)]
        @printf("  num_leaves=%2d lr=%.2f min_child=%2d -> grouped-CV MAE %.3f\n", nl, lr, mcs, mean(sc))
        mean(sc) < best[1] && (best = (mean(sc), hp))
    end
    hp = best[2]; println("selected: ", hp)

    # (b) 직접 대 2단계
    yd = max.(lgbm_reg(X[tr, :], y[tr], X[te, :]; hp..., base...), 0)
    clf = lgbm_clf_model(X[tr, :], Int.(y[tr] .> 0); hp..., base...); p_ins = lgbm_out(clf, X[te, :])
    pos = tr[y[tr] .> 0]
    reg = lgbm_reg_model(X[pos, :], y[pos]; hp..., base...); yr = max.(lgbm_out(reg, X[te, :]), 0)
    y2 = ifelse.(p_ins .> 0.5, yr, 0.0); ys = p_ins .* yr
    met, ins = y[te] .== 0, y[te] .> 0
    for (nm, yp) in (("direct regression", yd), ("two-stage (hard gate)", y2), ("two-stage (soft p*gap)", ys))
        @printf("%-26s MAE all %.3f | metals %.3f | non-metals %.3f | false-metal(gap>0 predicted 0) %d\n",
                nm, mae(yp, y[te]), mae(yp[met], y[te][met]), mae(yp[ins], y[te][ins]), sum((yp .== 0) .& ins))
    end
    @printf("classifier: accuracy %.3f  ROC-AUC %.3f  | regressor MAE on true non-metals %.3f\n",
            mean((p_ins .> 0.5) .== ins), roc_auc(Int.(ins), p_ins), mae(yr[ins], y[te][ins]))
    @printf("direct regression on metals: fraction predicted < 0.1 eV = %.2f\n", mean(yd[met] .< 0.1))

    # (c) 계열별
    println("family          n_test  mean_gap  MAE_direct  MAE_2stage")
    for f in ("oxide", "chalcogenide", "halide", "pnictide", "other")
        k = fam[te] .== f
        @printf("%-14s %6d %9.2f %11.3f %11.3f\n", f, sum(k), mean(y[te][k]), mae(yd[k], y[te][k]), mae(y2[k], y[te][k]))
    end

    # (d) 비금속 회귀기의 순열 중요도 (시험 비금속, 5회)
    ti = te[ins]; base_mae = mae(lgbm_out(reg, X[ti, :]), y[ti]); rng = MersenneTwister(0)
    imp = zeros(length(feat), 5)
    for j in 1:length(feat), r in 1:5
        Xp = X[ti, :]; Xp[:, j] = shuffle(rng, Xp[:, j]); imp[j, r] = mae(lgbm_out(reg, Xp), y[ti]) - base_mae
    end
    println("permutation importance (increase in MAE, eV), top 8:")
    for i in sortperm(-vec(mean(imp; dims = 2)))[1:8]
        @printf("   %-48s %.3f +/- %.3f\n", feat[i], mean(imp[i, :]), std(imp[i, :]; corrected = false))
    end
    println("LightGBM split-count top 5: ", feat[sortperm(-LightGBM.split_importance(reg))[1:5]])

    # (e) 등위 보정 (그룹 3겹, 겹마다 분류기 + 등위 회귀, 확률 평균)
    ybin = Float64.(y .> 0); p_cal = zeros(length(te))
    for (a, b) in group_kfold(g[tr], 3)
        c = lgbm_clf_model(X[tr[a], :], Int.(ybin[tr[a]]); hp..., base...)
        iso = isotonic_fit(lgbm_out(c, X[tr[b], :]), ybin[tr[b]])
        p_cal .+= isotonic_predict(iso, lgbm_out(c, X[te, :])) ./ 3
    end
    brier(p, t) = mean((p .- t) .^ 2)
    @printf("P(non-metal) calibration: Brier raw %.4f -> isotonic %.4f ; ECE raw %.4f -> isotonic %.4f\n",
            brier(p_ins, ins), brier(p_cal, ins), ece(p_ins, Float64.(ins)), ece(p_cal, Float64.(ins)))

    # (f) 구간: 분위 LightGBM, 그리고 RF 퍼짐의 분할 등각 재보정
    q = Dict(a => lgbm_reg(X[pos, :], y[pos], X[ti, :]; objective = "quantile", alpha = a, hp..., base...) for a in (0.05, 0.5, 0.95))
    @printf("quantile LGBM nominal 90%% interval: coverage %.3f, mean width %.2f eV\n",
            mean((y[ti] .>= q[0.05]) .& (y[ti] .<= q[0.95])), mean(q[0.95] .- q[0.05]))
    pa_, pc_ = group_shuffle_split(g[pos]; test_size = 0.25, rng = MersenneTwister(7))
    pa, pc = pos[pa_], pos[pc_]
    rf = rf_reg(X[pa, :], y[pa]; n_trees = 300, frac = 0.3, min_leaf = 2, seed = 0)
    spread(idx) = (T = tree_preds(rf, X[idx, :]); (vec(mean(T; dims = 1)), vec(std(T; dims = 1, corrected = false))))
    mu, sd = spread(ti); muc, sdc = spread(pc)
    z = abs.(y[ti] .- mu) ./ sd; zc = abs.(y[pc] .- muc) ./ sdc
    @printf("RF (fit on %d, calibrate on %d group-disjoint): MAE non-metals %.3f; mean sigma %.2f eV\n", length(pa), length(pc), mae(mu, y[ti]), mean(sd))
    for lev in (0.5, 0.68, 0.9, 0.95)
        qc = quantile(zc, lev)
        @printf("   nominal %.2f -> Gaussian(sigma) observed %.3f | conformal (k=%.2f) observed %.3f\n",
                lev, mean(z .<= quantile(Normal(), 0.5 + lev / 2)), qc, mean(z .<= qc))
    end
    @printf("Spearman(|error|, sigma) = %.3f\n", corspearman(abs.(y[ti] .- mu), sd))
    @printf("wall %.0f s\n", time() - t0)
end

main()
