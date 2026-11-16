# lab_8_12.jl — Lab 8.12: RTA 온도 제어기, 명목·무작위화·최악 대비 (labs/84_domain_rand.py)
# 가우스 선형 정책(관측 7개 + 편향), 시간별 배치 기저선의 REINFORCE, 손으로 쓴 Adam. numpy 판처럼 패키지 없이 옮겼다.
# 세 학습 방식 x 시드 3개, 각각 5 x 5 장비 격자와 무작위화 시험 2,000 에피소드로 평가한다.
include("ch8_common.jl")
include("furnace.jl")

"regime = :nominal이면 명목 장비, :dr이면 h ∈ [0.05, 0.2](로그 균일), G ∈ [0.16, 0.24], d ∈ {1, …, 4}"
function sample_plants(regime, n, rng)
    regime == :nominal && return NOM.h, NOM.G, NOM.d
    return exp.(log(0.05) .+ (log(0.2) - log(0.05)) .* rand(rng, n)), 0.16 .+ 0.08 .* rand(rng, n), rand(rng, 1:4, n)
end

"REINFORCE; cvar = α면 비용이 가장 큰 α 비율의 에피소드만으로 기울기를 낸다"
function train(regime, seed; iters = 700, n = 128, lr = 0.02, cvar = nothing)
    rng = Xoshiro(seed); th = zeros(8); opt = VAdam(8)
    for it in 0:iters-1
        sig = 0.15 * (1 - it / iters) + 0.03
        env = FurnaceBatch(n, sample_plants(regime, n, rng)..., rng)
        Phi, A, C = Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[]
        for t in 1:T_EP
            o, e = observe(env); f = feats(o); mu = vec(th' * f)
            a = mu .+ sig .* randn(rng, n)
            env.I = clamp.(env.I .+ e, -10, 10); env.e_prev = e
            push!(C, step!(env, a)); push!(Phi, f); push!(A, (a .- mu) ./ sig^2)     # ∇_θ log π = (a - μ)/σ² · φ
        end
        Cm = reduce(hcat, C)'                                                         # T x n
        G2 = -reverse(cumsum(reverse(Cm; dims = 1); dims = 1); dims = 1)               # 남은 수익(reward-to-go)
        tot = vec(sum(Cm; dims = 1))
        keep = cvar === nothing ? trues(n) : tot .>= quantile(tot, 1 - cvar)
        Gk = G2[:, keep]; adv = Gk .- mean(Gk; dims = 2); adv ./= std(adv; corrected = false) + 1e-8
        grad = zeros(8); kk = findall(keep)
        for t in 1:T_EP, (c, i) in enumerate(kk)
            grad .+= adv[t, c] * A[t][i] .* Phi[t][:, i]
        end
        adam_ascent!(opt, th, grad ./ length(kk), lr)
    end
    return th
end

const H_GRID, D_GRID = [0.05, 0.07, 0.1, 0.14, 0.2], [1, 2, 3, 4, 5]
for (name, reg, cv) in [("nominal", :nominal, nothing), ("DR", :dr, nothing), ("DR-CVaR(25%)", :dr, 0.25)]
    grids, ths, dr = Array{Float64}[], Vector{Float64}[], []
    for seed in 0:2
        th = train(reg, seed; cvar = cv); push!(ths, th)
        g = [mean(rollout(linpolicy(th), h, NOM.G, d, 200, Xoshiro(1000 + 10 * (i - 1) + (j - 1))).cost)
             for (i, h) in enumerate(H_GRID), (j, d) in enumerate(D_GRID)]
        push!(grids, g)
        rr = rollout(linpolicy(th), sample_plants(:dr, 2000, Xoshiro(99))..., 2000, Xoshiro(98))
        push!(dr, (mean = mean(rr.cost), p90 = quantile(rr.cost, 0.9), cvar25 = mean(sort(rr.cost)[end-499:end]), over = mean(rr.xmax .> 1.05)))
    end
    G = cat(grids...; dims = 3)                                                   # h x d x 시드
    worst = [maximum(G[:, 1:4, s]) for s in 1:3]
    @printf("%-13s nominal plant %5.1f +/- %4.1f K | h=0.05,d=4 %5.1f | h=0.2,d=1 %5.1f | worst of grid(d<=4) %5.1f +/- %4.1f K | grid d=5 worst %6.1f | DR-test mean %5.1f, p90 %5.1f, CVaR25 %5.1f K, overshoot>50K %4.1f%%\n",
            name, 1e3 * mean(G[3, 2, :]), 1e3 * std(G[3, 2, :]; corrected = false), 1e3 * mean(G[1, 4, :]), 1e3 * mean(G[5, 1, :]),
            1e3 * mean(worst), 1e3 * std(worst; corrected = false), 1e3 * mean(maximum(G[:, 5, s]) for s in 1:3),
            1e3 * mean(r.mean for r in dr), 1e3 * mean(r.p90 for r in dr), 1e3 * mean(r.cvar25 for r in dr), 100 * mean(r.over for r in dr))
    println("   mean grid (K), rows h=", join(H_GRID, ","), "; cols d=1..5")
    for i in 1:5
        println("     ", join([@sprintf("%6.1f", 1e3 * mean(G[i, j, :])) for j in 1:5], " "))
    end
    println("   theta (seed 0): ", join([@sprintf("%.3f", t) for t in ths[1]], " "))
    name == "nominal" && open(io -> println(io, join(ths[1], " ")), joinpath(@__DIR__, "84_theta_nominal_s0.txt"), "w")   # Lab 8.13의 전문가
end
