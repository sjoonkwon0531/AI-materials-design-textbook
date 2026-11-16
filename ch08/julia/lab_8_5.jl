# lab_8_5.jl — Lab 8.5: CIGS 조성 편집 환경, 보상 해킹과 두 겹의 처방 (labs/82_compenv.py + labs/82_comp_rl.py)
# 줄인 포트: Magpie 특징화(matminer)와 대리 모델 앙상블 학습, LightGBM 'second opinion'은 파이썬이 이미 계산해 둔 값을 읽는다
# (labs/82_grid_CuInGaSSe_N12.npz의 μ, σ, 5-NN 거리와 labs/82_comp_rl.npz의 LightGBM 예측). 그 위의 MDP, 환경 구조체
# (reset!/step!), 무작위·탐욕·Q-learning·BFS(T-공 안의 정확한 최적) 에이전트는 모두 옮겼다.
# 격자 상태의 순서는 파이썬 itertools.combinations(별과 막대)와 같아야 npz의 배열과 맞는다.
include("ch8_common.jl")
using NPZ

const ELEMS = ("Cu", "In", "Ga", "S", "Se")
const OXI = Dict("Cu" => 1, "In" => 3, "Ga" => 3, "S" => -2, "Se" => -2, "Te" => -2, "Zn" => 2, "Cd" => 2, "O" => -2)
const NAT, TARGET, TEP = 12, 1.4, 8

"0:(n-1)에서 r개를 고르는 조합, 사전식 순서 (itertools.combinations와 같음)"
function combinations_lex(n, r)
    out = Vector{Vector{Int}}(); c = collect(0:r-1)
    while true
        push!(out, copy(c))
        i = r
        while i >= 1 && c[i] == n - r + i - 1
            i -= 1
        end
        i == 0 && return out
        c[i] += 1
        for j in i+1:r
            c[j] = c[j-1] + 1
        end
    end
end
"합이 N이고 0이 아닌 성분이 둘 이상인 길이 k의 정수 벡터 (별과 막대)"
function grid_states(k, N)
    out = NTuple{5,Int}[]
    for c in combinations_lex(N + k - 1, k - 1)
        b = [-1; c; N + k - 1]
        v = ntuple(i -> b[i+1] - b[i] - 1, k)
        count(>(0), v) >= 2 && push!(out, v)
    end
    return out
end
formula(v) = join(["$(e)$(n)" for (e, n) in zip(ELEMS, v) if n > 0])
charge_balanced(v) = sum(OXI[e] * n for (e, n) in zip(ELEMS, v)) == 0

# ---------------------------------------------------------------- 대리 모델 표 (파이썬이 계산해 둔 값)
struct GridTable
    states::Vector{NTuple{5,Int}}
    index::Dict{NTuple{5,Int},Int}
    mu::Vector{Float64}
    sigma::Vector{Float64}
    dist::Vector{Float64}
    d95::Float64
    balanced::BitVector
end
function GridTable()
    st = grid_states(length(ELEMS), NAT)
    z = npzread(joinpath(CH8_LABS, "82_grid_CuInGaSSe_N12.npz"))
    @assert length(z["mu"]) == length(st)
    return GridTable(st, Dict(s => i for (i, s) in enumerate(st)), z["mu"], z["sigma"], z["dist"], only(z["d95"]),
                     BitVector(charge_balanced.(st)))
end
"s(x) = -|μ - target| - κσ - λ max(0, d/d95 - 1) - cb [전하 불균형]"
score(t::GridTable, target; kappa = 0.0, lam = 0.0, cb = 0.0) =
    -abs.(t.mu .- target) .- kappa .* t.sigma .- lam .* max.(0.0, t.dist ./ t.d95 .- 1) .- cb .* .!t.balanced

# ---------------------------------------------------------------- 환경 (Gymnasium CompositionEnv에 대응)
mutable struct CompositionEnv
    tab::GridTable
    S::Vector{Float64}                      # 퍼텐셜(점수)
    T::Int
    pairs::Vector{Tuple{Int,Int}}           # (i, j): 원자 하나를 i에서 j로; 마지막 (0, 0)은 "그대로"
    state::NTuple{5,Int}
    t::Int
    rng::Xoshiro
end
function CompositionEnv(tab, target, T; kappa = 0.0, lam = 0.0, cb = 0.0, seed = 0)
    k = length(ELEMS)
    pairs = [[(i, j) for i in 1:k for j in 1:k if i != j]; (0, 0)]
    return CompositionEnv(tab, score(tab, target; kappa, lam, cb), T, pairs, tab.states[1], 0, Xoshiro(seed))
end
nactions(env) = length(env.pairs)                           # 21
function next_state(env, s, a)
    i, j = env.pairs[a]
    (i == 0 || s[i] == 0) && return s
    v = collect(s); v[i] -= 1; v[j] += 1; v = Tuple(v)
    return haskey(env.tab.index, v) ? v : s                 # 원소가 하나만 남는 이동은 무효
end
function reset!(env; start = nothing)
    env.state = start === nothing ? rand(env.rng, env.tab.states) : start; env.t = 0
    return collect(env.state)
end
"반환: (관측, 보상, terminated, truncated). 보상은 퍼텐셜 차이라 수익 = s(x_T) - s(x_0)"
function step!(env, a)
    s0 = env.S[env.tab.index[env.state]]
    env.state = next_state(env, env.state, a); env.t += 1
    return collect(env.state), env.S[env.tab.index[env.state]] - s0, false, env.t >= env.T
end

# ---------------------------------------------------------------- 에이전트
"T번 편집 안의 정확한 최적 (너비 우선 탐색) — 퍼텐셜 기반 보상의 유한 지평 DP 답"
function ball_opt(env, S, start, T)
    frontier = Set([start]); seen = Set([start])
    for _ in 1:T
        nxt = Set{NTuple{5,Int}}()
        for s in frontier, a in 1:nactions(env)
            s2 = next_state(env, s, a)
            s2 ∉ seen && (push!(seen, s2); push!(nxt, s2))
        end
        frontier = nxt
    end
    return argmax(s -> S[env.tab.index[s]], collect(seen))
end
function run_policy(env, start, pol)
    s = start
    for t in 1:TEP
        s = next_state(env, s, pol(s))
    end
    return s
end
greedy(env, S) = s -> argmax(a -> S[env.tab.index[next_state(env, s, a)]], 1:nactions(env))
function qlearn(env, S; episodes = 20_000, alpha = 0.2, gamma = 0.95, seed = 0)
    rng = Xoshiro(seed); A = nactions(env); tab = env.tab
    Q = zeros(length(tab.states), A)
    for ep in 0:episodes-1
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        s = rand(rng, tab.states)
        for t in 1:TEP
            i = tab.index[s]
            a = rand(rng) < eps ? rand(rng, 1:A) : argmax(Q[i, :] .+ 1e-9 .* rand(rng, A))
            s2 = next_state(env, s, a); j = tab.index[s2]
            Q[i, a] += alpha * (S[j] - S[i] + gamma * maximum(Q[j, :]) - Q[i, a]); s = s2
        end
    end
    return Q
end

# ---------------------------------------------------------------- 실행
tab = GridTable()
second = npzread(joinpath(CH8_LABS, "82_comp_rl.npz"))["second"]   # LightGBM 예측 (독립 'second opinion')
St = reduce(vcat, [collect(s)' for s in tab.states])
tie = (St[:, 4] .== St[:, 5]) .& (St[:, 4] .> 0)                     # S = Se: Magpie의 'mode' 특징이 뒤집힌다
@printf("grid: %d compositions (k=5, N=%d); charge-balanced %d\n", length(tab.states), NAT, count(tab.balanced))
near = abs.(tab.mu .- TARGET) .< 0.05
@printf("|mu - %.1f| < 0.05 eV: %d compositions, of which charge-balanced %d\n", TARGET, count(near), count(near .& tab.balanced))
@printf("detectors of 'not charge balanced' (ROC-AUC): sigma %.2f, 5-NN distance %.2f, |MLP-LGBM| %.2f\n",
        auroc(.!tab.balanced, tab.sigma), auroc(.!tab.balanced, tab.dist), auroc(.!tab.balanced, abs.(tab.mu .- second)))

env = CompositionEnv(tab, TARGET, TEP; seed = 0)
o = reset!(env); @assert length(o) == 5 && sum(o) == NAT && all(0 .<= o .<= NAT)          # check_env에 해당하는 최소 점검
G = 0.0; s_init = env.state
for _ in 1:TEP
    global G += step!(env, rand(env.rng, 1:nactions(env)))[2]
end
@assert G ≈ env.S[tab.index[env.state]] - env.S[tab.index[s_init]]                          # 퍼텐셜 기반 보상의 망원합
println("environment contract: obs in space, $(nactions(env)) actions, return telescopes to s(x_T) - s(x_0)")

rng = Xoshiro(1)
starts = tab.states[randperm(rng, length(tab.states))[1:200]]
configs = [("naive", (;)), ("sigma-pen (k=2)", (kappa = 2.0,)), ("charge-bal", (cb = 1.0,)), ("charge-bal + sigma", (cb = 1.0, kappa = 2.0))]
println("\nT = $TEP edits from 200 random starts; columns: mean |mu-target|, charge-balanced fraction, mean sigma, mean |MLP-LGBM|, S=Se tie fraction")
for (name, kw) in configs
    S = score(tab, TARGET; kw...); env.S .= S
    g = argmax(S)
    @printf("[%s] global optimum of its score: %s mu=%.3f sigma=%.3f LGBM=%.2f\n", name, formula(tab.states[g]), tab.mu[g], tab.sigma[g], second[g])
    Q = qlearn(env, S)
    rows = [("random policy", [run_policy(env, s0, s -> rand(rng, 1:nactions(env))) for s0 in starts]),
            ("greedy (21 queries/step)", [run_policy(env, s0, greedy(env, S)) for s0 in starts]),
            ("Q-learning (20k ep.)", [run_policy(env, s0, s -> argmax(Q[tab.index[s], :])) for s0 in starts]),
            ("T-ball optimum (BFS)", [ball_opt(env, S, s0, TEP) for s0 in starts])]
    opt_sc = [S[tab.index[s]] for s in rows[end][2]]
    for (rname, finals) in rows
        idx = [tab.index[s] for s in finals]
        @printf("   %-26s err %.3f | bal %.2f | sigma %.3f | |MLP-LGBM| %.2f | tie %.2f | reaches optimum %.2f\n", rname,
                mean(abs.(tab.mu[idx] .- TARGET)), mean(tab.balanced[idx]), mean(tab.sigma[idx]),
                mean(abs.(tab.mu[idx] .- second[idx])), mean(tie[idx]), mean(isapprox.(S[idx], opt_sc)))
    end
    fin = rows[3][2]; u = unique(fin); c = [count(==(s), fin) for s in u]; o3 = sortperm(c; rev = true)[1:min(3, end)]
    println("   most frequent Q-learning end points: ", join(["$(formula(u[i]))($(c[i]))" for i in o3], ", "))
end
