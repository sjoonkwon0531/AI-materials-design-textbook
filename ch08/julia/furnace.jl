# furnace.jl — Lab 8.12, 8.13의 장난감 RTA 장비 (labs/84_thermal_env.py의 Julia 대응)
#   x_{t+1} = x_t + (G u_{t-d} - h x_t) + w_t,   u ∈ [0, 1],   x = (T - T_amb)/1000 K
# 명목: h = 0.1 /s, G = 0.2, d = 2, 잡음 0.003. 목표: 20 s 동안 0 → 1 선형, 그다음 40 s 유지 (60단계).
# 관측(7 x n, 열 = 에피소드): [e_t, e_{t-1}, I_t, x*_{t+1}, x*_{t+1} - x*_t, u_{t-1}, u_{t-2}]. 비용 |e| + 5 max(0, x - 1.05).
# n개의 에피소드를 한꺼번에 굴린다(에피소드마다 h, G, d가 다를 수 있다). 값은 모두 합성이다.
const T_EP, RAMP = 60, 20
const NOM = (h = 0.1, G = 0.2, d = 2)
target(t) = t < RAMP ? t / RAMP : 1.0

mutable struct FurnaceBatch
    n::Int
    h::Vector{Float64}
    G::Vector{Float64}
    d::Vector{Int}
    rng::Xoshiro
    noise::Float64
    x::Vector{Float64}
    ubuf::Matrix{Float64}        # ubuf[k, i] = 에피소드 i의 u_{t-k} (k = 1..8; 갱신 직후 k = 1이 방금의 u)
    t::Int
    I::Vector{Float64}           # 오차 적분 (±10에서 자름)
    e_prev::Vector{Float64}
end
asvec(v, n, T) = v isa Number ? fill(T(v), n) : collect(T.(v))
FurnaceBatch(n, h, G, d, rng; noise = 0.003) =
    FurnaceBatch(n, asvec(h, n, Float64), asvec(G, n, Float64), asvec(d, n, Int), rng, noise, zeros(n), zeros(8, n), 0, zeros(n), fill(target(0), n))

"관측 (7 x n)과 측정 오차 e (측정 잡음 포함)"
function observe(env::FurnaceBatch)
    xm = env.x .+ env.noise .* randn(env.rng, env.n)
    e = target(env.t) .- xm
    O = vcat(e', env.e_prev', env.I', fill(target(env.t + 1), 1, env.n), fill(target(env.t + 1) - target(env.t), 1, env.n),
             env.ubuf[1:1, :], env.ubuf[2:2, :])
    return O, e
end
"한 단계 진행, 단계 비용 (n) 반환"
function step!(env::FurnaceBatch, u)
    u = clamp.(u, 0.0, 1.0)
    env.ubuf = circshift(env.ubuf, (1, 0)); env.ubuf[1, :] = u
    ueff = [env.ubuf[env.d[i], i] for i in 1:env.n]                          # 파이썬 ubuf[:, d - 1]
    env.x = env.x .+ (env.G .* ueff .- env.h .* env.x) .+ env.noise .* randn(env.rng, env.n)
    env.t += 1
    return abs.(target(env.t) .- env.x) .+ 5.0 .* max.(0.0, env.x .- 1.05)
end
"policy(관측 7 x n) -> u (n). 반환: 에피소드별 평균 비용, 최대 x [, 관측 기록 (7 x n x T)]"
function rollout(policy, h, G, d, n, rng; T = T_EP, record = false, noise = 0.003)
    env = FurnaceBatch(n, h, G, d, rng; noise)
    C, xmax = zeros(n), zeros(n); O = record ? zeros(7, n, T) : nothing
    for t in 1:T
        o, e = observe(env); u = policy(o)
        env.I = clamp.(env.I .+ e, -10, 10); env.e_prev = e
        C .+= step!(env, u); xmax = max.(xmax, env.x)
        record && (O[:, :, t] = o)
    end
    return (cost = C ./ T, xmax = xmax, obs = O)
end

const SCALE = [1, 1, 0.1, 1, 10, 1, 1, 1.0]                              # 특징 척도(적분 /10, 목표 기울기 x10), 편향
feats(o) = vcat(o, ones(1, size(o, 2))) .* SCALE                         # 8 x n
linpolicy(th) = o -> vec(th' * feats(o))
