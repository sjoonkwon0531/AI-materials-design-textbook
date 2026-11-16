# lab_8_2.jl — Lab 8.2: 결함 어닐링 스케줄, 동적 계획법으로 정확히 풀고 표본으로 배운다
# (labs/81_anneal_env.py + labs/81_anneal.py). 패키지 없이 옮겼다. 상태 n = 0..20은 Julia에서 1..21 칸이다.
# 끝에 (e) 강건 가치 반복(Derivation Box 8.12)을 Julia 판에만 덧붙였다.
include("ch8_common.jl")

# ---------------------------------------------------------------- 환경 (출생–소멸 Markov 사슬)
const kB = 8.617333e-5
const NDEF = 20
const TEMPS = [500.0, 700.0, 900.0, 1100.0]
const E_M, E_F = 0.40, 0.12
const NU = 0.8 / exp(-E_M / (kB * 1100.0))           # k_ann(1100 K) = 0.8
const C0, GAMMA = 0.01, 0.95
const nS, nA = NDEF + 1, length(TEMPS)

function rates(T; ka_mult = 1.0, kg_mult = 1.0)
    ka = NU * exp(-E_M / (kB * T))
    return ka_mult * ka, kg_mult * ka * exp(-E_F / (kB * T))
end
reward(n, a) = -(n / NDEF) - C0 * (TEMPS[a] - 300) / 800           # n은 결함 수(0..20)

"P[s, s′, a]와 R[s, a] (s = n + 1)"
function build_model(; ka_mult = 1.0, kg_mult = 1.0)
    P = zeros(nS, nS, nA); R = zeros(nS, nA)
    for a in 1:nA
        ka, kg = rates(TEMPS[a]; ka_mult, kg_mult)
        for n in 0:NDEF
            s = n + 1; pd = ka * n / NDEF; pu = kg * (NDEF - n) / NDEF
            n > 0 && (P[s, s-1, a] = pd)
            n < NDEF && (P[s, s+1, a] = pu)
            P[s, s, a] = 1 - (n > 0 ? pd : 0.0) - (n < NDEF ? pu : 0.0)
            R[s, a] = reward(n, a)
        end
    end
    @assert all(sum(P; dims = 2) .≈ 1) && all(P .>= 0)
    return P, R
end

"모델 없는 에이전트가 보는 시뮬레이터: (다음 상태 칸, 보상)"
function env_step(s, a, rng)
    n = s - 1; ka, kg = rates(TEMPS[a])
    r = reward(n, a); u = rand(rng)
    pd = ka * n / NDEF; pu = kg * (NDEF - n) / NDEF
    n2 = u < pd ? n - 1 : (u < pd + pu ? n + 1 : n)
    return n2 + 1, r
end

const P, R = build_model()
q_from_v(V; Pm = P, Rm = R) = Rm .+ GAMMA .* reduce(hcat, [Pm[:, :, a] * V for a in 1:nA])
"정책 평가 = 선형계 (I - γ P_π) V = R_π"
function policy_eval_exact(pi; Pm = P, Rm = R)
    Ppi = reduce(vcat, [Pm[s, :, pi[s]]' for s in 1:nS]); Rpi = [Rm[s, pi[s]] for s in 1:nS]
    return (I - GAMMA * Ppi) \ Rpi
end
greedy_policy(Q) = [argmax(Q[s, :]) for s in 1:nS]

# ---------------------------------------------------------------- (a) 가치 반복, (b) 정책 반복
function value_iteration(; tol = 1e-12, kmax = 5000)
    V = zeros(nS); hist = Float64[]
    for k in 1:kmax
        Vn = vec(maximum(q_from_v(V); dims = 2))
        push!(hist, maximum(abs.(Vn .- V))); V = Vn
        hist[end] < tol && break
    end
    return V, hist
end
function policy_iteration()
    pi = ones(Int, nS); it = 0
    while true
        V = policy_eval_exact(pi); it += 1
        pin = greedy_policy(q_from_v(V))
        pin == pi && return pi, V, it
        pi = pin
    end
end

# ---------------------------------------------------------------- (c) MC 대 TD(0), (d) Q-learning
function sample_episode(pi, s0, H, rng)
    s = s0; traj = Tuple{Int,Float64,Int}[]
    for t in 1:H
        s2, r = env_step(s, pi[s], rng); push!(traj, (s, r, s2)); s = s2
    end
    return traj
end
"every-visit MC: t < t_use인 방문만 쓴다(각 수익이 H - t_use 단계 이상을 덮도록)"
function mc_eval(pi, n_ep, H, rng; t_use = 50)
    S, C = zeros(nS), zeros(nS)
    for _ in 1:n_ep
        traj = sample_episode(pi, rand(rng, 1:nS), H, rng); g = 0.0
        for t in H:-1:1
            s, r, _ = traj[t]; g = r + GAMMA * g
            t <= t_use && (S[s] += g; C[s] += 1)
        end
    end
    return [C[s] > 0 ? S[s] / C[s] : 0.0 for s in 1:nS]
end
function td0_eval(pi, n_ep, H, rng; alpha = 0.1)
    V = zeros(nS)
    for _ in 1:n_ep, (s, r, s2) in sample_episode(pi, rand(rng, 1:nS), H, rng)
        V[s] += alpha * (r + GAMMA * V[s2] - V[s])           # 시간 제한 절단: 언제나 부트스트랩
    end
    return V
end
function q_learning(n_steps, rng; eps = 0.2, H = 50, Qstar = nothing, log_every = 0)
    Q, C = zeros(nS, nA), zeros(nS, nA); log = []
    s = rand(rng, 1:nS); t_ep = 0
    for t in 1:n_steps
        a = rand(rng) < eps ? rand(rng, 1:nA) : argmax(Q[s, :])
        s2, r = env_step(s, a, rng)
        C[s, a] += 1; alpha = 1 / C[s, a]^0.6
        Q[s, a] += alpha * (r + GAMMA * maximum(Q[s2, :]) - Q[s, a])
        s = s2; t_ep += 1
        t_ep == H && (s = rand(rng, 1:nS); t_ep = 0)             # 탐색적 시작으로 재시작
        if log_every > 0 && t % log_every == 0
            loss = minimum(policy_eval_exact(greedy_policy(Q)) .- policy_eval_exact(greedy_policy(Qstar)))
            push!(log, (t, maximum(abs.(Q .- Qstar)), loss))
        end
    end
    return Q, log
end

# ---------------------------------------------------------------- 실행
println("temperatures (K): ", Int.(TEMPS), "  gamma = $GAMMA  N = $NDEF")
for T in TEMPS
    ka, kg = rates(T)
    @printf("  T=%6.0f K  k_ann=%.3f  k_gen=%.4f  equilibrium n = %.2f\n", T, ka, kg, NDEF * kg / (ka + kg))
end
Vs, hist = value_iteration()
Qs = q_from_v(Vs); pis = greedy_policy(Qs)
ratios = hist[2:end] ./ hist[1:end-1]
@printf("\n(a) value iteration: %d sweeps to |dV|<1e-12; successive ratio -> %.4f (k=50), %.4f (k=200); gamma = %.2f\n",
        length(hist), ratios[51], ratios[201], GAMMA)
println("    optimal T*(n) for n=0..20: ", Int.(TEMPS[pis]))
println("    V*(n) at n=0,5,10,20: ", round.(Vs[[1, 6, 11, 21]]; digits = 3))
pi_pi, V_pi, it = policy_iteration()
@printf("(b) policy iteration: %d evaluations; same policy as VI: %s; max|V_PI - V_VI| = %.1e\n",
        it, pi_pi == pis, maximum(abs.(V_pi .- Vs)))
for (name, a) in [("always 1100 K", 4), ("always 900 K", 3), ("always 700 K", 2)]
    Vf = policy_eval_exact(fill(a, nS))
    @printf("    fixed %-13s: V(20) = %.3f   (optimal %.3f)\n", name, Vf[21], Vs[21])
end

println("\n(c) policy evaluation of pi*: episodes of H=150 steps, random start, 20 seeds")
Vp = policy_eval_exact(pis)
for n_ep in (20, 200), (name, f) in [("MC", rng -> mc_eval(pis, n_ep, 150, rng)),
                                     ("TD0 a=0.1", rng -> td0_eval(pis, n_ep, 150, rng; alpha = 0.1)),
                                     ("TD0 a=0.3", rng -> td0_eval(pis, n_ep, 150, rng; alpha = 0.3))]
    v = reduce(hcat, [f(Xoshiro(1000 + sd)) .- Vp for sd in 0:19])         # nS x 20
    @printf("    %4d episodes  %-10s: RMSE = %.3f, bias = %.3f, sd across seeds = %.3f\n", n_ep, name,
            sqrt(mean(abs2, v)), mean(abs.(mean(v; dims = 2))), mean(std(v; dims = 2, corrected = false)))
end

println("\n(d) Q-learning, eps=0.2 behaviour, alpha=1/N(s,a)^0.6, 5 seeds")
srt = sort(Qs; dims = 2); gap = srt[:, end] .- srt[:, end-1]
@printf("    action gap Q*(best)-Q*(2nd): smallest %.4f at n=%d, median %.4f\n", minimum(gap), argmin(gap) - 1, median(gap))
for sd in 0:4
    Q, log = q_learning(400_000, Xoshiro(sd); Qstar = Qs, log_every = 20_000)
    gq = greedy_policy(Q)
    @printf("    seed %d: max|Q-Q*| = %.3f, greedy policy matches pi* in %d/%d states, worst-state value loss = %.4f, mismatched n = %s\n",
            sd, maximum(abs.(Q .- Qs)), count(gq .== pis), nS, max(0.0, -log[end][3]), string(findall(gq .!= pis) .- 1))
end

# ---------------------------------------------------------------- (e) 강건 가치 반복 (Julia 판에만; Derivation Box 8.12)
# 불확실성 집합: (s, a)마다 네 '꼭짓점' 전이 행(k_ann × {0.7, 1}, k_gen × {1, 1.5})의 볼록 껍질.
# 선형 함수 Σ p V의 최소는 꼭짓점에서 나오므로 안쪽 최소화는 네 행 중 최솟값이다. (s, a)마다 따로 고르므로 직사각형 집합이다.
const VERTS = [build_model(; ka_mult = k1, kg_mult = k2)[1] for k1 in (0.7, 1.0) for k2 in (1.0, 1.5)]
function robust_vi(; tol = 1e-12, kmax = 5000)
    V = zeros(nS); hist = Float64[]
    for k in 1:kmax
        Q = R .+ GAMMA .* reduce(hcat, [minimum(reduce(hcat, [Pv[:, :, a] * V for Pv in VERTS]); dims = 2) for a in 1:nA])
        Vn = vec(maximum(Q; dims = 2))
        push!(hist, maximum(abs.(Vn .- V))); V = Vn
        hist[end] < tol && return V, greedy_policy(Q), hist
    end
end
Vr, pir, hr = robust_vi()
@printf("\n(e) robust VI (rectangular set, 4 vertex plants): %d sweeps; ratio at k=200 %.4f\n", length(hr), hr[202] / hr[201])
println("    robust T(n) for n=0..20: ", Int.(TEMPS[pir]))
# 묶인(non-rectangular) 불확실성: 장비 하나(꼭짓점 하나)를 모든 상태에 고정했을 때의 최악
coupled(pi) = minimum(policy_eval_exact(pi; Pm = Pv)[21] for Pv in VERTS)
@printf("    V(20): robust (rectangular) %.3f | worst single plant for pi_rob %.3f | for pi* %.3f | pi* nominal %.3f\n",
        Vr[21], coupled(pir), coupled(pis), Vs[21])
