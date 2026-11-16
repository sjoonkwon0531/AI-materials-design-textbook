# lab_8_8.jl — Lab 8.8: 두 공정 에이전트의 조정 실패, 'climbing game' (labs/82_marl.py)
# 에이전트 1은 전구체 경로(A/B/C), 에이전트 2는 어닐링 레시피(a/b/c)를 고르고 같은 보상을 받는다. 패키지 없이 옮겼다.
include("ch8_common.jl")

const RG = [11.0 -30.0 0.0; -30.0 7.0 6.0; 0.0 0.0 5.0]      # 공유 보상; 최적 (A, a) = 11
const NOISE = 1.0
eps_at(t, T) = max(0.02, 1.0 - t / (0.7 * T))

"독립 학습자: 각자 상대를 환경의 일부로 본다"
function independent(seed; T = 3000, alpha = 0.1, lenient = false)
    rng = Xoshiro(seed); Q1, Q2 = zeros(3), zeros(3)
    for t in 0:T-1
        e = eps_at(t, T)
        a = rand(rng) < e ? rand(rng, 1:3) : argmax(Q1)
        b = rand(rng) < e ? rand(rng, 1:3) : argmax(Q2)
        r = RG[a, b] + NOISE * randn(rng)
        for (Q, k) in ((Q1, a), (Q2, b))
            d = r - Q[k]
            Q[k] += (lenient && d <= 0 ? 0.1 * alpha : alpha) * d   # lenient: 오를 땐 빠르게, 내릴 땐 느리게
        end
    end
    return argmax(Q1), argmax(Q2)
end
"중앙 학습자: 9개의 결합 행동 위의 Q"
function joint(seed; T = 3000, alpha = 0.1)
    rng = Xoshiro(seed); Q = zeros(3, 3)
    for t in 0:T-1
        if rand(rng) < eps_at(t, T)
            a, b = rand(rng, 1:3), rand(rng, 1:3)
        else
            a, b = Tuple(argmax(Q))
        end
        Q[a, b] += alpha * (RG[a, b] + NOISE * randn(rng) - Q[a, b])
    end
    return Tuple(argmax(Q))
end

nm = "ABC"
for (label, fn) in [("independent Q-learners", s -> independent(s)), ("independent, lenient update", s -> independent(s; lenient = true)),
                    ("centralized joint-action Q", joint)]
    res = [fn(s) for s in 0:199]; u = unique(res); c = [count(==(x), res) for x in u]; o = sortperm(c; rev = true)
    println(rpad(label, 28), ": ", join(["($(nm[u[i][1]]),$(lowercase(nm[u[i][2]])))=$(@sprintf("%.2f", c[i] / 200))" for i in o[1:min(4, end)]], ", "),
            @sprintf(" | mean converged payoff %.2f", mean(RG[a, b] for (a, b) in res)))
end
println("expected payoff of agent-1 actions vs a uniformly random partner: ", round.(vec(mean(RG; dims = 2)); digits = 2))
