# lab_7_3.jl — Lab 7.3: Magpie 밴드갭 위의 최적화기 비교와 AdamW의 차이 (labs/71_optim.py)
# 분할은 파이썬과 같게 하려고 6장 Lab 6.13의 KFold(seed 0) 겹 0과 그 안의 검증 분할을 인덱스 CSV로 읽는다.
# 최적화기: Optimisers.jl의 Descent/Momentum/Nesterov/AdaGrad/RMSProp/Adam과 ch7_common.jl의 TorchAdamW.
# torch RMSprop의 alpha 기본값은 0.99이므로 RMSProp(η, 0.99)로 맞췄다. bf16 autocast는 Julia에 대응이 없어
# 숫자 형식의 범위만 보인다. L-BFGS는 Optim.jl(강한 Wolfe 선 탐색).
include("ch7_common.jl")
include("flux_tools.jl")
using Optim, LineSearches, BFloat16s
import LightGBM

X, y, _ = load_magpie()
tr, te = read_idx("fold0_train.csv"), read_idx("fold0_test.csv")
tr_in, va = read_idx("fold0_inner_train.csv"), read_idx("fold0_val.csv")
P = fit_prep(X[tr_in, :])
Xa, Xv, Xt = feats(P, X[tr_in, :]), feats(P, X[va, :]), feats(P, X[te, :])
ya, yv, yt = Float32.(y[tr_in]), Float32.(y[va]), Float32.(y[te])
Xf, yf = hcat(Xa, Xv), vcat(ya, yv)
@printf("n=%d, p=%d -> %d after variance filter; fold-0 train %d / val %d / test %d\n",
        length(y), size(X, 2), length(P.keep), length(tr_in), length(va), length(te))

mlp(seed) = mlp_chain(Xoshiro(seed), [size(Xa, 1), 256, 128, 1])

"rule을 lr0로 시작해 epochs 동안 학습; sched(step, total)는 lr0에 곱할 배율. 반환: (망, 에폭별 평가 MAE)"
function fit(rule, seed; epochs = 80, bs = 64, sched = nothing, Xtr = Xa, ytr = ya, Xev = Xv, yev = yv)
    rng = Xoshiro(seed)
    net = mlp(seed); opt = Flux.setup(rule, net)
    lr0 = rule.eta; n = length(ytr); total = epochs * cld(n, bs); step = 0
    curve = Float64[]
    for ep in 1:epochs
        for idx in batches(rng, n, bs)
            sched === nothing || Flux.adjust!(opt, lr0 * sched(step, total))
            xb, yb = Xtr[:, idx], ytr[idx]
            g = only(Flux.gradient(m -> mean(abs2, vec(m(xb)) .- yb), net))
            Flux.update!(opt, net, g); step += 1
        end
        push!(curve, gap_mae(net, Xev, yev))
    end
    return net, curve
end

const OPTS = ["SGD" => (lr -> Optimisers.Descent(lr), [3e-3, 1e-2, 3e-2]),
              "SGD+momentum" => (lr -> Optimisers.Momentum(lr, 0.9), [1e-3, 3e-3, 1e-2]),
              "Nesterov" => (lr -> Optimisers.Nesterov(lr, 0.9), [1e-3, 3e-3, 1e-2]),
              "Adagrad" => (lr -> Optimisers.AdaGrad(lr, 1e-10), [3e-3, 1e-2, 3e-2]),
              "RMSProp" => (lr -> Optimisers.RMSProp(lr, 0.99, 1e-8), [1e-4, 3e-4, 1e-3]),
              "Adam" => (lr -> Optimisers.Adam(lr), [3e-4, 1e-3, 3e-3]),
              "AdamW(wd=1e-2)" => (lr -> TorchAdamW(lr; lambda = 1e-2), [3e-4, 1e-3, 3e-3])]

function partA()
    println("=== Part A: optimizers (80 epochs, batch 64, constant LR; LR by min val MAE; seeds 0-2) ===")
    for (name, (mk, lrs)) in OPTS
        best = (lr = 0.0, v = Inf, c = Float64[])
        for lr in lrs
            _, c = fit(mk(lr), 0)
            minimum(c) < best.v && (best = (lr = lr, v = minimum(c), c = c))
        end
        tests = [last(fit(mk(best.lr), s; Xtr = Xf, ytr = yf, Xev = Xt, yev = yt)[2]) for s in 0:2]
        reach = something(findfirst(<(0.5), best.c), -1)
        @printf("%-15s %8.0e %8.3f %22.3f +/- %.3f %14d\n", name, best.lr, best.v, mean(tests), std(tests; corrected = false), reach)
    end
end

"Part B(i): 매개변수 하나, 평균 0인 잡음 기울기 σ ξ. 데이터 기울기만 주고 감쇠는 규칙이 처리"
function toy(kind, sigma; steps = 2000, lr = 1e-3, lam = 0.5, seed = 0)
    rng = Xoshiro(seed); th = [1.0]
    rule = kind == :L2 ? AdamL2(lr, lam) : TorchAdamW(lr; lambda = lam)
    st = Optimisers.setup(rule, th)
    for _ in 1:steps
        st, th = Optimisers.update!(st, th, [sigma * randn(rng)])
    end
    return abs(th[1])
end

function partB()
    @printf("=== Part B(i): theta0 = 1, lr 1e-3, lambda 0.5, 2000 steps (AdamW decay alone: %.3f) ===\n", exp(-1e-3 * 0.5 * 2000))
    for sg in (0.0, 1e-3, 1e-1, 1.0, 10.0)
        @printf("sigma %8.0e   Adam+L2 |theta| %8.3f   AdamW |theta| %8.3f\n", sg,
                mean(toy(:L2, sg; seed = s) for s in 0:19), mean(toy(:W, sg; seed = s) for s in 0:19))
    end
    println("=== Part B(ii): Adam(lr 1e-3) + coupled L2 vs AdamW (decoupled), test MAE (seed 0) ===")
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        row = String[]
        for (kind, rule) in (("Adam+L2", AdamL2(1e-3, lam)), ("AdamW", TorchAdamW(1e-3; lambda = lam)))
            net, c = fit(rule, 0; Xtr = Xf, ytr = yf, Xev = Xt, yev = yt)
            wn = sqrt(sum(sum(abs2, l.weight) for l in net.layers if l isa Dense))
            push!(row, @sprintf("%s: MAE %.3f  ||W|| %6.1f", kind, last(c), wn))
        end
        @printf("lambda %7.0e   %s\n", lam, join(row, "   |   "))
    end
end

function partC()
    println("=== Part C: schedules, L-BFGS, number formats ===")
    _, cA = fit(TorchAdamW(1e-3; lambda = 1e-2), 0; Xtr = Xf, ytr = yf, Xev = Xt, yev = yt)
    _, cB = fit(TorchAdamW(3e-3; lambda = 1e-2), 0; sched = warm_cos, Xtr = Xf, ytr = yf, Xev = Xt, yev = yt)
    @printf("AdamW constant lr 1e-3            : final %.3f (min %.3f; final-5 std %.4f)\n", last(cA), minimum(cA), std(cA[end-4:end]; corrected = false))
    @printf("AdamW warmup(5%%)+cosine, peak 3e-3: final %.3f (min %.3f; final-5 std %.4f)\n", last(cB), minimum(cB), std(cB[end-4:end]; corrected = false))
    # L-BFGS: 평탄화한 매개변수 벡터 위에서 Optim.jl. torch와 달리 바깥 반복마다 이력이 새로 시작한다
    θ, re = Flux.destructure(mlp(0))
    f(t) = mean(abs2, vec(re(t)(Xf)) .- yf)
    g!(G, t) = (G .= only(Zygote.gradient(f, t)))
    lcur = Float64[]
    for it in 1:40
        res = Optim.optimize(f, g!, θ, LBFGS(; m = 20, linesearch = LineSearches.StrongWolfe()),
                             Optim.Options(; iterations = 20))
        θ = Optim.minimizer(res); push!(lcur, gap_mae(re(θ), Xt, yt))
    end
    @printf("L-BFGS full batch (40 x 20 it): test MAE after 200 it %.3f, after 800 it %.3f, min %.3f; train MAE %.3f\n",
            lcur[10], lcur[end], minimum(lcur), gap_mae(re(θ), Xf, yf))
    h = [1e-8, 1e-5, 3e-4]
    println("fp16 underflow: ", Float64.(Float16.(h)), " | scaled by 2^16: ", Float64.(Float16.(h .* 65536)),
            " | unscaled in fp32: ", [@sprintf("%.3e", Float32(Float16(v * 65536)) / 65536) for v in h])
    println("bf16 keeps range: ", [@sprintf("%.3e", Float32(BFloat16(Float32(v)))) for v in h])
    est = LightGBM.LGBMRegression(; num_iterations = 600, learning_rate = 0.05, num_leaves = 31, num_threads = 1, verbosity = -1)
    LightGBM.fit!(est, X[tr, :], y[tr]; verbosity = -1)
    p = vec(LightGBM.predict(est, X[te, :]; verbosity = -1))
    @printf("reference LightGBM on fold 0: test MAE %.3f, clipped at 0: %.3f\n", mae(p, y[te]), mae(max.(p, 0), y[te]))
end

partA(); partB(); partC()
