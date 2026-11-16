# film_env.jl — Lab 8.14, 8.15의 박막 성장 환경 (labs/85_film_env.py의 Julia 대응)
# 고리 위 L개 자리의 높이 h(x, t)에 대한 선형 확률 성장 방정식
#     ∂h/∂t = F - κ(T) ∂⁴h/∂x⁴ - a ∂²h/∂x² + η
# 실수 푸리에 기저에서 각 모드 c_m은 독립 Ornstein–Uhlenbeck 과정이고(Γ_m = κ q̃⁴ - a q̃², q̃² = 4 sin²(π m/L)),
# 온도가 일정한 한 증착 간격 동안 정확히 적분된다:  c ← c e^{-Γ dt} + √(F(1 - e^{-2Γ dt})/(2Γ)) ξ.
# 행동 a ∈ [-1, 1] → 설정 온도 700 + 100a K. 실제 온도 = 설정 + δ(에피소드마다 U(-40, 40) K, 숨음) + N(0, 3²) K.
# 관측(5): [t/N, log(w + 1e-3), log(w_low + 1e-3), log(w_high + 1e-3), 직전 행동]. 보상 -(w_{t+1} - w_t) - λ tb_t.
const KB_EV = 8.617333e-5

Base.@kwdef struct FilmParams
    L::Int = 128
    N::Int = 12
    kappa_ref::Float64 = 1.0
    Es::Float64 = 1.0
    a::Float64 = 0.25
    Etb::Float64 = 3.0
    lam::Float64 = 0.02
    Tref::Float64 = 700.0
    delta::Float64 = 40.0
    jitter::Float64 = 3.0
    m_split::Int = 8
end
"모드 번호 m (cos 1..L/2-1, sin 1..L/2-1, Nyquist L/2), q̃², 저주파 마스크"
function modes(P::FilmParams)
    m = [1:P.L÷2-1; 1:P.L÷2-1; P.L ÷ 2]
    return m, 4 .* sin.(pi .* m ./ P.L) .^ 2, m .<= P.m_split
end
kappa(P, T) = P.kappa_ref .* exp.(-(P.Es / KB_EV) .* (1 ./ T .- 1 / P.Tref))
thermal_budget(P, T) = exp.(-(P.Etb / KB_EV) .* (1 ./ T .- 1 / P.Tref))
to_T(a) = 700.0 .+ 100.0 .* clamp.(a, -1.0, 1.0)

"모든 모드의 정확한 OU 갱신. c: 모드 x 배치, T: 배치"
function ou_step(c, T, P, q2, rng; dt = 1.0, F = 1.0)
    G = q2 .^ 2 .* kappa(P, T)' .- P.a .* q2                       # 모드 x 배치
    e = exp.(-G .* dt)
    var = @. ifelse(abs(G * dt) < 1e-8, F * dt, F * (1 - e^2) / (2 * ifelse(abs(G * dt) < 1e-8, 1.0, G)))
    return c .* e .+ sqrt.(var) .* randn(rng, size(c))
end
rough(c, P) = vec(sqrt.(sum(abs2, c; dims = 1) ./ P.L))
rough(c, P, mask) = vec(sqrt.(sum(abs2, c[mask, :]; dims = 1) ./ P.L))
"관측 (5 x 배치)"
function film_obs(c, t, last, P, low)
    n = size(c, 2)
    return vcat(fill(t / P.N, 1, n), log.(rough(c, P) .+ 1e-3)', log.(rough(c, P, low) .+ 1e-3)', log.(rough(c, P, .!low) .+ 1e-3)', last')
end

# ---------------------------------------------------------------- Gymnasium FilmGrowthEnv에 대응하는 구조체
mutable struct FilmGrowthEnv
    P::FilmParams
    q2::Vector{Float64}
    low::BitVector
    fixed_delta::Union{Nothing,Float64}
    open_loop::Bool              # true면 시계 t/N만 보인다 (OpenLoop 감싸개)
    rng::Xoshiro
    c::Matrix{Float64}
    delta::Float64
    t::Int
    last_a::Float64
    w::Float64
    TB::Float64
end
function FilmGrowthEnv(; P = FilmParams(), delta = nothing, open_loop = false, seed = 0)
    _, q2, low = modes(P)
    return FilmGrowthEnv(P, q2, low, delta, open_loop, Xoshiro(seed), zeros(length(q2), 1), 0.0, 0, 0.0, 0.0, 0.0)
end
function observation(env)
    o = Float32.(vec(film_obs(env.c, env.t, [env.last_a], env.P, env.low)))
    env.open_loop && (o[2:end] .= 0)
    return o
end
function reset!(env; seed = nothing)
    seed === nothing || (env.rng = Xoshiro(seed))
    env.delta = env.fixed_delta === nothing ? env.P.delta * (2rand(env.rng) - 1) : env.fixed_delta
    env.c = zeros(length(env.q2), 1); env.t = 0; env.last_a = 0.0; env.w = 0.0; env.TB = 0.0
    return observation(env)
end
"반환: (관측, 보상, terminated, truncated, info)"
function step!(env, action)
    a = clamp(Float64(action[1]), -1.0, 1.0)
    T = to_T(a) + env.delta + env.P.jitter * randn(env.rng)
    env.c = ou_step(env.c, [T], env.P, env.q2, env.rng)
    tb = thermal_budget(env.P, T); wn = rough(env.c, env.P)[1]
    r = -(wn - env.w) - env.P.lam * tb
    env.w = wn; env.TB += tb; env.last_a = a; env.t += 1
    return observation(env), r, env.t >= env.P.N, false, (T = T, tb = tb, w = wn, TB = env.TB)
end

"""n개 에피소드를 공통 난수로 한꺼번에 평가. schedule(t, obs 5 x n) -> 행동 n개 (t는 0부터).
반환 (J, w_N, TB[, 온도 n x N, δ])"""
function rollout_batch(schedule, n, seed; P = FilmParams(), delta = nothing, return_traj = false)
    rng = Xoshiro(seed); _, q2, low = modes(P)
    d = delta === nothing ? P.delta .* (2 .* rand(rng, n) .- 1) : fill(Float64(delta), n)
    c = zeros(length(q2), n); TB = zeros(n); last = zeros(n); Ts = zeros(n, P.N)
    for t in 0:P.N-1
        a = clamp.(vec(schedule(t, film_obs(c, t, last, P, low))), -1, 1)
        T = to_T(a) .+ d .+ P.jitter .* randn(rng, n)
        c = ou_step(c, T, P, q2, rng)
        TB .+= thermal_budget(P, T); last = a; Ts[:, t+1] = T
    end
    wN = rough(c, P); J = -wN .- P.lam .* TB
    return return_traj ? (J, wN, TB, Ts, d) : (J, wN, TB)
end
