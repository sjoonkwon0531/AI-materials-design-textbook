# lab_7_18.jl — Lab 7.18: 149개 조성 특성 위의 MLP 대 LightGBM (labs/76_mlp_bandgap.py)
# 6장 Lab 6.16과 같은 겹(무작위 5겹 seed 0, 화학계별 GroupKFold 5겹)은 파이썬이 내보낸 시험 인덱스로 읽는다.
# MLP 149-256-256-256-1(SiLU, Dropout 0.1), TorchAdamW(2e-3, wd 1e-4) + one-cycle(10% warmup), L1 손실,
# 조기 종료(인내 15), 겹마다 K = 4개 앙상블 → 예측 퍼짐. MLUtils.DataLoader로 미니배치, LightGBM.jl로 기준선.
# 겹 안의 검증 분할(train_test_split)은 Julia 난수로 뽑으므로 숫자는 통계적으로만 일치한다. CPU 한 스레드에서 수십 분.
include("ch7_common.jl")
include("flux_tools.jl")
import LightGBM

function make_mlp(rng, p; h = 256, depth = 3, drop = 0.1)
    layers = Any[]; d = p
    for _ in 1:depth
        push!(layers, tdense(rng, d, h, swish), Dropout(drop)); d = h
    end
    return Chain(layers..., tdense(rng, d, 1))
end

function fit_one(Xtr, ytr, Xva, yva, seed; epochs = 80, bs = 128, lr = 2e-3, wd = 1e-4, patience = 15)
    rng = Xoshiro(seed); model = make_mlp(rng, size(Xtr, 1))
    opt = Flux.setup(TorchAdamW(lr; lambda = wd), model)
    loader = MLUtils.DataLoader((Xtr, ytr); batchsize = bs, shuffle = true, rng = rng)
    total = epochs * length(loader); step = 0
    best, best_m, wait, ep_best = Inf, deepcopy(model), 0, 0
    for ep in 1:epochs
        for (xb, yb) in loader                                             # Dropout은 gradient 안에서만 켜진다
            Flux.adjust!(opt, onecycle_lr(lr, step, total; pct = 0.1)); step += 1
            g = only(Flux.gradient(m -> mean(abs.(vec(m(xb)) .- yb)), model))
            Flux.update!(opt, model, g)
        end
        v = gap_mae(model, Xva, yva)
        if v < best - 1e-4
            best, best_m, wait, ep_best = v, deepcopy(model), 0, ep
        else
            wait += 1
            wait >= patience && break
        end
    end
    return best_m, ep_best
end

function main()
    X, y, chemsys = load_features149()
    n = length(y)
    @printf("n = %d, p = %d, metals %.1f%%\n", n, size(X, 2), 100 * mean(y .== 0))
    K = 4
    for (name, tag) in (("random 5-fold (seed 0)", "random"), ("group 5-fold (chemsys)", "group"))
        o_gb, o_mu, o_sd, o_one = zeros(n), zeros(n), zeros(n), zeros(n); used = Int[]
        for f in 0:4
            te = read_idx("cv_$(tag)_fold$(f).csv"); tr = setdiff(1:n, te)
            est = LightGBM.LGBMRegression(; num_iterations = 400, learning_rate = 0.05, num_leaves = 31, min_data_in_leaf = 10,
                                          bagging_fraction = 0.8, bagging_freq = 1, feature_fraction = 0.5, num_threads = 1, verbosity = -1)
            LightGBM.fit!(est, X[tr, :], y[tr]; verbosity = -1)
            o_gb[te] = max.(vec(LightGBM.predict(est, X[te, :]; verbosity = -1)), 0)
            preds = zeros(K, length(te))
            for k in 0:K-1
                p = shuffle(Xoshiro(100 * f + k), tr); nv = round(Int, 0.1 * length(tr))
                iva, itr = p[1:nv], p[nv+1:end]
                P = fit_prep(X[itr, :]; thr = nothing, clipv = 6.0)          # 학습 부분으로만 맞춘 StandardScaler
                m, eb = fit_one(feats(P, X[itr, :]), Float32.(y[itr]), feats(P, X[iva, :]), Float32.(y[iva]), 100 * f + k)
                push!(used, eb)
                preds[k+1, :] = max.(vec(m(feats(P, X[te, :]))), 0)
            end
            o_one[te] = preds[1, :]; o_mu[te] = vec(mean(preds; dims = 1)); o_sd[te] = vec(std(preds; dims = 1))
            @printf("  [%s] fold %d: LGBM %.3f | MLP single %.3f | MLP ens(%d) %.3f eV\n", tag, f, mae(o_gb[te], y[te]), mae(o_one[te], y[te]), K, mae(o_mu[te], y[te]))
        end
        err = abs.(o_mu .- y); nm = y .> 0
        q = quantile(o_sd, [0.2, 0.8])
        println(name, ":")
        @printf("   LightGBM MAE %.3f | non-metals %.3f\n", mae(o_gb, y), mae(o_gb[nm], y[nm]))
        @printf("   MLP single MAE %.3f | non-metals %.3f\n", mae(o_one, y), mae(o_one[nm], y[nm]))
        @printf("   MLP ensemble of %d MAE %.3f | non-metals %.3f;  average with LGBM %.3f\n", K, mae(o_mu, y), mae(o_mu[nm], y[nm]), mae((o_mu .+ o_gb) ./ 2, y))
        @printf("   early-stopped epoch: median %d (range %d-%d)\n", round(Int, median(used)), minimum(used), maximum(used))
        @printf("   spread vs |error|: Pearson r = %.2f; MAE in lowest/highest spread quintile %.3f / %.3f eV\n",
                cor(o_sd, err), mean(err[o_sd .<= q[1]]), mean(err[o_sd .>= q[2]]))
        @printf("   coverage of mu +/- 1.96 sd: %.1f%% (nominal 95%%)\n", 100 * mean(err .<= 1.96 .* o_sd))
    end
end

main()
