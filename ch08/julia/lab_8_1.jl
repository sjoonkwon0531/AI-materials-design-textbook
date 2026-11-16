# lab_8_1.jl — Lab 8.1: 도펀트 후보 10종의 순차 스크리닝, 탐욕·ε-탐욕·UCB (labs/81_bandit.py)
# R = 2000개의 독립 문제를 함께 푼다. 파이썬은 (R, K) 배열로 벡터화했고, Julia는 문제마다 도는 루프가 그대로 빠르다.
# 동점 깨기는 파이썬과 같은 1e-9 크기의 무작위 잡음, 표본 평균 갱신 Q ← Q + (r - Q)/N.
include("ch8_common.jl")

const K, T, R = 10, 1000, 2000
const MU = randn(Xoshiro(81), K, R)                 # 문제마다 참 평균 μ_k ~ N(0, 1) (열 = 문제)
const BEST = vec(maximum(MU; dims = 1))
const ASTAR = [argmax(MU[:, r]) for r in 1:R]

function run_bandit(method, param; seed = 1)
    rng = Xoshiro(seed)
    Q = zeros(K, R); N = zeros(K, R)
    rew, opt, regret = zeros(T), zeros(T), zeros(T)
    for t in 1:T, r in 1:R
        q, n = view(Q, :, r), view(N, :, r)
        if method == :eps
            a = rand(rng) < param ? rand(rng, 1:K) : argmax(q .+ 1e-9 .* rand(rng, K))
        else                                         # UCB1 꼴: 안 뽑은 팔 먼저, 그다음 Q + c √(ln t / N)
            score = [n[k] == 0 ? Inf : q[k] + param * sqrt(log(t) / n[k]) for k in 1:K]
            a = argmax(score .+ 1e-9 .* rand(rng, K))
        end
        x = MU[a, r] + randn(rng)
        n[a] += 1
        q[a] += (x - q[a]) / n[a]
        rew[t] += x / R; opt[t] += (a == ASTAR[r]) / R; regret[t] += (BEST[r] - MU[a, r]) / R
    end
    return rew, opt, cumsum(regret)
end

println("K=$K arms, T=$T pulls, R=$R problems, mu_k~N(0,1), noise sd=1")
@printf("%-18s %20s %18s %16s\n", "method", "avg reward(last100)", "%optimal(last100)", "cum.regret@1000")
for (m, p, name) in [(:eps, 0.0, "greedy (eps=0)"), (:eps, 0.01, "eps-greedy 0.01"),
                     (:eps, 0.1, "eps-greedy 0.1"), (:ucb, 2.0, "UCB c=2")]
    rew, opt, creg = run_bandit(m, p)
    @printf("%-18s %20.3f %17.1f%% %16.1f\n", name, mean(rew[end-99:end]), 100 * mean(opt[end-99:end]), creg[end])
end
@printf("mean of best arm over problems = %.3f\n", mean(BEST))
