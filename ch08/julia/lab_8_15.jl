# lab_8_15.jl — Lab 8.15: 과거 공정 기록 위의 대리 모델, AdamW·SAM·ASAM·앙상블·신뢰 영역 (labs/85_sam_surrogate.py)
# 모형 기반 최적화: 과거 실행(650–720 K 근처의 12단계 열린 고리 일정, 잡음 섞인 에피소드 하나씩) -> 대리 모델 Ĵ(일정)
# -> 상자 [600, 800]^12 전체에서 Ĵ를 최대화 -> 제안된 일정을 참 시뮬레이터(2,000 에피소드)에서 평가.
# 대리 모델은 Flux MLP(12-64-64-1, SiLU), SAM·ASAM은 sam.jl의 두 번 계산 갱신, 최대화는 Optimisers.Adam의 사영 상승.
include("ch8_common.jl")
include(joinpath(CH7_JL, "ch7_common.jl"))           # TorchAdamW (PyTorch식 AdamW)
include(joinpath(CH7_JL, "flux_tools.jl"))           # mlp_chain, batches
include("sam.jl")
include("film_env.jl")
using Flux

const P = FilmParams()
mlp(seed) = mlp_chain(Xoshiro(seed), [P.N, 64, 64, 1], swish)                 # swish = SiLU

"method = :none(AdamW), :sam, :asam. 전체 400 에폭, 배치 64, 학습률 일정 없음"
function train(X, y, seed; method = :none, rho = 0.05, epochs = 400, lr = 3e-3, wd = 1e-4)
    θ, re = Flux.destructure(mlp(seed)); opt = Optimisers.setup(TorchAdamW(lr; lambda = wd), θ)
    Xt, yt = Float32.(X), Float32.(y); rng = Xoshiro(seed); ss = SAMState()
    for ep in 1:epochs, i in batches(rng, length(yt), 64)
        opt, θ, _ = sam_update!(opt, θ, q -> mean(abs2, vec(re(q)(Xt[:, i])) .- yt[i]), ss; rho, mode = method)
    end
    return re(θ)
end
function lam_of(net, X, y)
    θ, L = chain_mse(net, X, y)                          # Float64, 정확한 HVP용 손실
    return lambda_max(L, θ; iters = 40, tol = 0.0)[1]
end
"64개의 무작위 시작점에서 사영 Adam 상승 300번; 예측이 가장 큰 점을 고른다"
function maximise(score, lo, hi, seed; n_start = 64, steps = 300, lr = 0.03)
    x = lo .+ (hi .- lo) .* rand(Xoshiro(seed), Float32, P.N, n_start)
    opt = Optimisers.setup(Optimisers.Adam(lr), x)
    for _ in 1:steps
        g = only(Zygote.gradient(z -> -sum(score(z)), x))
        opt, x = Optimisers.update!(opt, x, g); x .= clamp.(x, lo, hi)
    end
    s = score(x)
    return x[:, argmax(s)], maximum(s)
end
true_J(sched; n = 2000, seed = 777) = mean(rollout_batch((t, o) -> fill(sched[t+1], size(o, 2)), n, seed; P)[1])

# ---------------------------------------------------------------- '과거 실행' 데이터
rng = Xoshiro(0); n_tr = 400
base = (650 .+ 70 .* rand(rng, n_tr))' .+ 10 .* randn(rng, P.N, n_tr)            # K, 일정 x 실행
X = (base .- 700) ./ 100                                                          # 행동 단위
y = [rollout_batch((t, o) -> fill(X[t+1, i], 1), 1, 10_000 + i - 1; P)[1][1] for i in 1:n_tr]
ymu, ysd = mean(y), std(y; corrected = false); yn = (y .- ymu) ./ ysd
Xid = ((650 .+ 70 .* rand(rng, 300))' .+ 10 .* randn(rng, P.N, 300) .- 700) ./ 100   # 분포 안
Xood = 2 .* rand(rng, P.N, 300) .- 1                                                # 상자 전체
Jid = [true_J(Xid[:, i]; n = 200, seed = 5) for i in 1:300]; Jood = [true_J(Xood[:, i]; n = 200, seed = 5) for i in 1:300]
@printf("training runs: %d; label J range [%.2f, %.2f]; data T range [%.0f, %.0f] K; best training label %.3f\n",
        n_tr, minimum(y), maximum(y), minimum(base), maximum(base), maximum(y))
@printf("reference: tuned two-level schedule true J = %.3f\n", true_J([fill(-0.3, 6); fill(-0.1, 6)]))
@printf("reference: best historical run re-run 2,000 times: true J = %.3f\n", true_J(X[:, argmax(y)]))

lo_box, hi_box = -ones(Float32, P.N), ones(Float32, P.N)
lo_tr, hi_tr = Float32.(vec(minimum(X; dims = 2))), Float32.(vec(maximum(X; dims = 2)))      # 신뢰 영역: 데이터의 단계별 범위
function evaluate(nets, score, lo, hi, seed)
    f = Z -> mean(vec(n(Float32.(Z))) for n in nets) .* ysd .+ ymu
    x_opt, _ = maximise(score, lo, hi, seed)
    pred = f(reshape(x_opt, :, 1))[1]; tj = true_J(x_opt)
    return (mae_id = mean(abs.(f(Xid) .- Jid)), mae_ood = mean(abs.(f(Xood) .- Jood)), pred = pred, true_ = tj, gap = pred - tj,
            dist = 100 * minimum(norm(X[:, i] .- x_opt) for i in 1:n_tr), Tmin = minimum(to_T(x_opt)), Tmax = maximum(to_T(x_opt)))
end
methods = ["AdamW", "SAM rho=0.05", "SAM rho=0.2", "ASAM rho=0.5", "Ensemble-5 LCB", "AdamW + trust region",
           "SAM rho=0.2 + trust region", "Ensemble LCB + trust region"]
R = Dict(m => [] for m in methods); LAM = Dict(m => Float64[] for m in methods[1:4])
for seed in 0:4
    nets = Dict("AdamW" => train(X, yn, seed), "SAM rho=0.05" => train(X, yn, seed; method = :sam, rho = 0.05),
                "SAM rho=0.2" => train(X, yn, seed; method = :sam, rho = 0.2), "ASAM rho=0.5" => train(X, yn, seed; method = :asam, rho = 0.5))
    for m in methods[1:4]
        push!(LAM[m], lam_of(nets[m], X, yn))
        push!(R[m], evaluate([nets[m]], z -> vec(nets[m](z)), lo_box, hi_box, seed))
    end
    ens = [nets["AdamW"]; [train(X, yn, 100 * seed + k) for k in 1:4]]
    lcb = z -> begin                                                          # 하한 신뢰 경계 μ - 2σ (구성원 5개)
        Pm = reduce(vcat, [n(z) for n in ens])
        vec(mean(Pm; dims = 1) .- 2 .* std(Pm; dims = 1))
    end
    push!(R["Ensemble-5 LCB"], evaluate(ens, lcb, lo_box, hi_box, seed))
    push!(R["AdamW + trust region"], evaluate([nets["AdamW"]], z -> vec(nets["AdamW"](z)), lo_tr, hi_tr, seed))
    push!(R["SAM rho=0.2 + trust region"], evaluate([nets["SAM rho=0.2"]], z -> vec(nets["SAM rho=0.2"](z)), lo_tr, hi_tr, seed))
    push!(R["Ensemble LCB + trust region"], evaluate(ens, lcb, lo_tr, hi_tr, seed))
    println("  seed $seed done")
end
println("\nmean +/- sd over 5 seeds; J in units of the reward (higher is better); dist = nearest training run [K]")
@printf("%-28s %8s %7s %8s %7s %14s %12s %6s %18s\n", "surrogate / optimiser", "lam_max", "MAE id", "MAE ood", "pred J", "true J", "gap", "dist", "T range of optimum")
for m in methods
    a = k -> [getfield(q, k) for q in R[m]]
    lam = haskey(LAM, m) ? @sprintf("%8.2f", mean(LAM[m])) : lpad("-", 8)
    @printf("%-28s %s %7.3f %8.2f %7.2f %7.2f+/-%4.2f %6.2f+/-%4.2f %6.0f   %4.0f-%4.0f K\n", m, lam, mean(a(:mae_id)), mean(a(:mae_ood)),
            mean(a(:pred)), mean(a(:true_)), std(a(:true_)), mean(a(:gap)), std(a(:gap)), mean(a(:dist)), mean(a(:Tmin)), mean(a(:Tmax)))
end
