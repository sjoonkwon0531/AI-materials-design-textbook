# lab_8_14.jl — Lab 8.14: 숨은 열 접촉 아래의 박막 성장, PPO 대 조율된 일정과 무작위 탐색 (labs/85_ppo.py)
# 환경은 film_env.jl의 구조체(reset!/step!), 기준선(조율된 상수·2단 일정, 무작위 탐색, δ를 아는 오라클)은 벡터화한 rollout_batch.
# PPO는 처음부터 썼다: Flux 행위자–비평가(5-64-64-1 tanh 두 개 + log σ), GAE(λ = 0.95, γ = 1), 잘린 대리 목적,
# 전역 기울기 노름 0.5 자르기, Adam 3e-4, 환경 16개 x 48단계 x 312회 갱신 = 240k 단계. 시드 3개 + 열린 고리 3개.
include("ch8_common.jl")
include(joinpath(CH7_JL, "flux_tools.jl"))            # tdense (PyTorch식 초기화)
include("film_env.jl")
using Flux, Optimisers, Zygote

const P = FilmParams()
const TEST_SEED, N_TEST = 12345, 2000

env = FilmGrowthEnv(); o1 = reset!(env; seed = 7); o2 = reset!(FilmGrowthEnv(); seed = 7)
println("environment contract: obs ", length(o1), " x Float32, action in [-1, 1]; same seed -> same first obs: ", o1 == o2)

# ---------------------------------------------------------------- 기준선 (학습용 시드로 조율)
constpol(Tset) = (t, o) -> fill((Tset - 700) / 100, size(o, 2))
J_of(fn; n = 600, seed = 1, kw...) = mean(rollout_batch(fn, n, seed; P, kw...)[1])
Ts = 620:5:780
T_const = Ts[argmax([J_of(constpol(T)) for T in Ts])]
twolevel(k, a, b) = (t, o) -> fill(((t < k ? a : b) - 700) / 100, size(o, 2))
_, (k2, T1, T2) = findmax(Dict((k, a, b) => J_of(twolevel(k, a, b); n = 300) for k in (3, 6, 9) for a in 640:10:740 for b in 640:10:760))
Tact = Ts[argmax([J_of(constpol(T); delta = 0.0) for T in Ts])]                 # δ = 0에서 가장 좋은 실제 온도
println("tuned constant set-point $T_const K | tuned two-level ($k2 steps at $T1 K, then $T2 K) | oracle actual T $Tact K")
rs_rng = Xoshiro(3); cands = 2 .* rand(rs_rng, 2500, P.N) .- 1                  # 무작위 탐색: 2,500 후보 x 10 에피소드
scores = [mean(rollout_batch((t, o) -> fill(cands[i, t+1], size(o, 2)), 10, 100 + i - 1; P)[1]) for i in 1:2500]
rs_best = cands[argmax(scores), :]
println("random search: best training score ", @sprintf("%.3f", maximum(scores)), "; schedule (K) ", join(round.(Int, to_T(rs_best)), " "))

# ---------------------------------------------------------------- PPO
struct ActorCritic{A,V}
    pi::A
    v::V
    log_std::Vector{Float32}
end
Flux.@layer ActorCritic
function ActorCritic(rng; n_obs = 5, h = 64)
    last = tdense(rng, h, 1); last.weight .*= 0.01f0; last.bias .= 0                 # 처음 정책 평균 ≈ 0
    return ActorCritic(Chain(tdense(rng, n_obs, h, tanh), tdense(rng, h, h, tanh), last),
                       Chain(tdense(rng, n_obs, h, tanh), tdense(rng, h, h, tanh), tdense(rng, h, 1)), Float32[-0.7])
end
normal_logpdf(a, mu, ls) = -0.5f0 .* ((a .- mu) ./ exp(ls)) .^ 2 .- ls .- 0.9189385f0     # 0.9189… = ½ log 2π

function ppo(seed; open_loop = false, total_steps = 240_000, n_env = 16, n_roll = 48, epochs = 8, mb = 256, lr = 3e-4,
             gamma = 1.0f0, lam = 0.95f0, clip = 0.2f0, max_norm = 0.5f0)
    rng = Xoshiro(seed); ac = ActorCritic(rng)
    θ, re = Flux.destructure(ac); opt = Optimisers.setup(Optimisers.Adam(lr), θ)
    envs = [FilmGrowthEnv(; open_loop) for _ in 1:n_env]
    obs = reduce(hcat, [reset!(envs[i]; seed = 1000 * seed + i - 1) for i in 1:n_env])   # 5 x n_env
    ep_ret = zeros(n_env); done_rets = Float64[]; curve = []
    n_upd = total_steps ÷ (n_env * n_roll)
    for u in 1:n_upd
        ac = re(θ); O, Aa, LP, Rw, D, V = [], [], [], [], [], []
        for t in 1:n_roll
            mu = vec(ac.pi(obs)); a = mu .+ exp(ac.log_std[1]) .* randn(rng, Float32, n_env)
            push!(O, obs); push!(Aa, a); push!(LP, normal_logpdf(a, mu, ac.log_std[1])); push!(V, vec(ac.v(obs)))
            r, d, nxt = zeros(Float32, n_env), zeros(Float32, n_env), similar(obs)
            for i in 1:n_env                                                        # SAME_STEP 자동 재시작
                o2, ri, term, trunc, _ = step!(envs[i], a[i])
                r[i] = ri; ep_ret[i] += ri
                if term || trunc
                    d[i] = 1; push!(done_rets, ep_ret[i]); ep_ret[i] = 0; o2 = reset!(envs[i])
                end
                nxt[:, i] = o2
            end
            push!(Rw, r); push!(D, d); obs = nxt
        end
        v_last = vec(ac.v(obs))
        adv = zeros(Float32, n_roll, n_env); g = zeros(Float32, n_env)              # GAE(λ); 끝난 단계는 부트스트랩 없음
        for t in n_roll:-1:1
            v_next = t == n_roll ? v_last : V[t+1]
            delta = Rw[t] .+ gamma .* v_next .* (1 .- D[t]) .- V[t]
            g = delta .+ gamma * lam .* (1 .- D[t]) .* g; adv[t, :] = g
        end
        RET = vec((adv .+ reduce(hcat, V)')')                                       # 시간 순서대로 이어 붙임
        ADV = vec(adv'); ADV = (ADV .- mean(ADV)) ./ (std(ADV) + 1f-8)
        Ox, Ax, LPx = reduce(hcat, O), reduce(vcat, Aa), reduce(vcat, LP)
        n = length(ADV)
        for _ in 1:epochs
            perm = randperm(rng, n)
            for s in 1:mb:n
                i = perm[s:min(s + mb - 1, n)]
                g_ = only(Zygote.gradient(θ) do q
                    m = re(q); mu = vec(m.pi(Ox[:, i]))
                    ratio = exp.(normal_logpdf(Ax[i], mu, m.log_std[1]) .- LPx[i])
                    l_clip = -mean(min.(ratio .* ADV[i], clamp.(ratio, 1 - clip, 1 + clip) .* ADV[i]))
                    l_clip + 0.5f0 * mean(abs2, vec(m.v(Ox[:, i])) .- RET[i])
                end)
                gn = norm(g_); gn > max_norm && (g_ = g_ .* (max_norm / gn))            # torch clip_grad_norm_ (전역 노름)
                opt, θ = Optimisers.update!(opt, θ, g_)
            end
        end
        (u % 50 == 0 || u == n_upd) && push!(curve, (u * n_env * n_roll, mean(done_rets[max(1, end - 399):end])))
    end
    return re(θ), curve
end
policy_fn(ac) = (t, o) -> vec(ac.pi(Float32.(o)))                                  # 결정론적: 평균 행동

# ---------------------------------------------------------------- 공통 시험 시드로 평가
results = Dict{String,Any}()
function report(name, fn)
    J, w, TB, T, d = rollout_batch(fn, N_TEST, TEST_SEED; P, return_traj = true)
    results[name] = (J = J, w = w, TB = TB, T = T, d = d)
    @printf("  %-30s J = %7.3f +/- %.3f | w_N %.3f | TB %6.2f | 5th pct J %7.3f\n", name, mean(J), std(J) / sqrt(length(J)),
            mean(w), mean(TB), quantile(J, 0.05))
    return mean(J)
end
@printf("\nTEST: %d episodes, seed %d, delta ~ U(-%.0f,%.0f) K, N = %d ML\n", N_TEST, TEST_SEED, P.delta, P.delta, P.N)
report("tuned constant", constpol(T_const))
report("tuned two-level", twolevel(k2, T1, T2))
report("random search (25k episodes)", (t, o) -> fill(rs_best[t+1], size(o, 2)))
ppoJ = [report("PPO seed $s", policy_fn(ppo(s)[1])) for s in 0:2]
@printf("  PPO over 3 training seeds: J = %.3f +/- %.3f (sd over seeds)\n", mean(ppoJ), std(ppoJ))
olJ = map(0:2) do s                                                                # 막을 보지 못하는 같은 PPO
    f = policy_fn(ppo(s; open_loop = true)[1])
    report("PPO open-loop seed $s", (t, o) -> f(t, vcat(o[1:1, :], zeros(4, size(o, 2)))))
end
@printf("  PPO open-loop over 3 seeds: J = %.3f +/- %.3f\n", mean(olJ), std(olJ))
d_test = P.delta .* (2 .* rand(Xoshiro(TEST_SEED), N_TEST) .- 1)                   # rollout_batch와 같은 첫 난수
report("oracle (knows delta)", (t, o) -> (Tact .- d_test .- 700) ./ 100)
for s in 0:2                                                                       # PPO는 막을 온도계로 읽는가
    r = results["PPO seed $s"]; Tset = vec(mean((r.T .- r.d)[:, 5:end]; dims = 2))
    slope = cov(r.d, Tset) / var(r.d)
    @printf("  PPO seed %d: corr(set-point over steps 5-12, delta) = %+.2f; slope dT_set/d delta = %+.2f\n", s, cor(Tset, r.d), slope)
end
