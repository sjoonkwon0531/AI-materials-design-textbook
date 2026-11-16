# lab_8_3.jl — Lab 8.3: 공정 창의 가장자리, SARSA 대 Q-learning, 그리고 죽음의 삼조 (labs/81_cliff_triad.py)
# (a) 절벽 걷기(4 x 12 격자; 행 1이 위, 행 4가 출발·절벽·목표 줄) — 표 형식, 패키지 없음.
# (b) Baird의 반례: 기대(DP식) 준경사 TD 갱신. 세 다리 중 하나를 빼면 수렴한다. 결정론적 계산이다.
include("ch8_common.jl")

# ---------------------------------------------------------------- (a) 절벽 걷기
const HH, WW = 4, 12
const START, GOAL = (4, 1), (4, 12)
const MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # 위, 오른쪽, 아래, 왼쪽

function env_step(s, a)
    r, c = s; dr, dc = MOVES[a]
    r2, c2 = clamp(r + dr, 1, HH), clamp(c + dc, 1, WW)
    (r2 == 4 && 2 <= c2 <= 11) && return START, -100.0, false   # 절벽에서 떨어짐: 출발점으로
    return (r2, c2), -1.0, (r2, c2) == GOAL
end
eps_greedy(Q, s, eps, rng) = rand(rng) < eps ? rand(rng, 1:4) : argmax_tie(rng, Q[s[1], s[2], :])

function run_cliff(method; n_ep = 500, eps = 0.1, alpha = 0.5, seed = 0)
    rng = Xoshiro(seed); Q = zeros(HH, WW, 4); rets = zeros(n_ep)
    for ep in 1:n_ep
        s = START; a = eps_greedy(Q, s, eps, rng); G = 0.0
        for t in 1:10_000
            s2, r, done = env_step(s, a); G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            nxt = done ? 0.0 : (method == :sarsa ? Q[s2[1], s2[2], a2] : maximum(Q[s2[1], s2[2], :]))
            Q[s[1], s[2], a] += alpha * (r + nxt - Q[s[1], s[2], a])
            s, a = s2, a2
            done && break
        end
        rets[ep] = G
    end
    return Q, rets
end
function greedy_path(Q)
    s = START; path = [s]
    for _ in 1:100
        s, _, done = env_step(s, argmax(Q[s[1], s[2], :])); push!(path, s)
        done && return path
    end
    return nothing                                         # 탐욕 정책이 고리를 돈다
end

println("(a) cliff walking, eps=0.1, alpha=0.5, gamma=1, 500 episodes, 50 runs")
for m in (:sarsa, :qlearn)
    Rm = zeros(50, 500); plen = Int[]
    for sd in 0:49
        Q, rets = run_cliff(m; seed = sd); Rm[sd+1, :] = rets
        p = greedy_path(Q); push!(plen, p === nothing ? -1 : length(p) - 1)
    end
    last = vec(mean(Rm[:, 401:500]; dims = 2))
    L = filter(>(0), plen); u = sort(unique(L))
    @printf("  %-7s: online return (episodes 401-500) = %7.2f +- %.2f;  greedy path length: %s (counts %s), greedy policy loops in %d runs\n",
            m, mean(last), std(last) / sqrt(50), string(u), string([count(==(k), L) for k in u]), count(<(0), plen))
end
for m in (:sarsa, :qlearn)
    Q, _ = run_cliff(m; seed = 0); p = greedy_path(Q)
    p === nothing || @printf("  seed-0 greedy path of %-7s: rows visited %s, length %d\n", m, string(sort(unique(first.(p)))), length(p) - 1)
end

# ---------------------------------------------------------------- (b) Baird의 반례
"mode = :triad(세 요소 모두), :mc(부트스트래핑 없음: 목표 = 참 이득 0), :tabular(함수 근사 없음)"
function baird(mode; sweeps = 5000, alpha = 0.01, gamma = 0.99)
    X = zeros(7, 8)
    for i in 1:6
        X[i, i] = 2; X[i, 8] = 1
    end
    X[7, 7] = 1; X[7, 8] = 2
    mode == :tabular && (X = Matrix{Float64}(I, 7, 7))
    w = ones(size(X, 2)); mode == :tabular || (w[7] = 10.0)
    norms, vmax = Float64[], Float64[]
    for k in 1:sweeps
        v = X * w
        target = mode == :mc ? zeros(7) : fill(gamma * v[7], 7)   # E_π[R + γ v(s′)] = γ v(7)
        w = w .+ (alpha / 7) .* (X' * (target .- v))            # 균일 상태 가중(행동 정책의 분포)
        push!(norms, norm(w)); push!(vmax, maximum(abs.(X * w)))
    end
    return norms, vmax
end
println("\n(b) Baird's counterexample, expected semi-gradient updates, alpha=0.01, gamma=0.99")
for mode in (:triad, :mc, :tabular)
    n, vm = baird(mode)
    @printf("  %-8s: ||w|| at sweep 1/1000/5000 = %.3g / %.3g / %.3g;  max|v - v_pi| = %.3g / %.3g / %.3g\n",
            mode, n[1], n[1000], n[end], vm[1], vm[1000], vm[end])
end
