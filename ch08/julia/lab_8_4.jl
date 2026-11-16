# lab_8_4.jl — Lab 8.4: REINFORCE, 기저선, A2C, 그리고 경사 추정량의 분산 (labs/81_pg.py)
# Gymnasium의 CartPole-v1 대신 같은 운동 방정식(오일러 적분, τ = 0.02 s, 500단계 절단)의 Julia 환경 구조체를
# reset!/step!으로 썼다. 파이썬 환경을 그대로 몰고 싶으면 이 절 본문의 PythonCall 예를 보라.
# 정책·가치 망은 Flux.jl(4-32-32, tanh), 기울기는 Zygote, 최적화는 Optimisers.Adam. 학습 15회 + 분산 실험으로 수 분 걸린다.
include("ch8_common.jl")
include(joinpath(CH7_JL, "flux_tools.jl"))            # tdense, mlp_chain (PyTorch식 초기화)
using Flux, Optimisers, Zygote

# ---------------------------------------------------------------- CartPole-v1과 같은 동역학
mutable struct CartPole
    s::Vector{Float64}          # (x, ẋ, θ, θ̇)
    t::Int
    rng::Xoshiro
end
CartPole(seed) = CartPole(zeros(4), 0, Xoshiro(seed))
function reset!(env::CartPole)
    env.s = rand(env.rng, 4) .* 0.1 .- 0.05; env.t = 0
    return Float32.(env.s)
end
"반환: (관측, 보상, terminated, truncated). 넘어진 단계도 보상 1 (gymnasium과 같음)"
function step!(env::CartPole, a::Int)                  # a ∈ {0, 1}
    g, mc, mp, l, fmag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x, xd, th, thd = env.s
    f = a == 1 ? fmag : -fmag
    ct, st = cos(th), sin(th)
    temp = (f + mp * l * thd^2 * st) / (mc + mp)
    thacc = (g * st - ct * temp) / (l * (4 / 3 - mp * ct^2 / (mc + mp)))
    xacc = temp - mp * l * thacc * ct / (mc + mp)
    env.s = [x + tau * xd, xd + tau * xacc, th + tau * thd, thd + tau * thacc]
    env.t += 1
    term = abs(env.s[1]) > 2.4 || abs(env.s[3]) > 12 * 2pi / 360
    return Float32.(env.s), 1.0, term, env.t >= 500
end

const GAMMA, LAM = 0.99, 0.95
mlp(out, rng) = mlp_chain(rng, [4, 32, 32, out], tanh)

"에피소드 하나: S (4 x T, Float32), A (0/1), 보상, terminated, 마지막 상태"
function rollout(pol, seed; rng = Xoshiro(seed))
    env = CartPole(seed); s = reset!(env); S, A, Rw = Vector{Float32}[], Int[], Float64[]
    while true
        z = pol(s); p1 = 1 / (1 + exp(z[1] - z[2]))           # 두 행동의 softmax
        a = Int(rand(rng) < p1)
        s2, r, term, trunc = step!(env, a)
        push!(S, s); push!(A, a); push!(Rw, r); s = s2
        (term || trunc) && return reduce(hcat, S), A, Rw, term, s2
    end
end
function reward_to_go(rw; gamma = GAMMA, bootstrap = 0.0)
    G = similar(rw); g = bootstrap
    for t in length(rw):-1:1
        g = rw[t] + gamma * g; G[t] = g
    end
    return G
end
"GAE: A_t = Σ_l (γλ)^l δ_{t+l}"
function gae(rw, v, v_last; lam = LAM, gamma = GAMMA)
    vn = [v[2:end]; v_last]; delta = rw .+ gamma .* vn .- v
    A = similar(rw); a = 0.0
    for t in length(rw):-1:1
        a = delta[t] + gamma * lam * a; A[t] = a
    end
    return A
end
"Σ_t w_t log π(a_t | s_t)"
logp_sum(pol, S, A, w) = sum(Float32.(w) .* logsoftmax(pol(S))[CartesianIndex.(A .+ 1, 1:length(A))])

function train(method, seed; n_ep = 300, lr_pi = 3e-3, lr_v = 1e-2)
    rng = Xoshiro(seed)
    pol, V = mlp(2, rng), mlp(1, rng)
    opi, ov = Optimisers.setup(Optimisers.Adam(lr_pi), pol), Optimisers.setup(Optimisers.Adam(lr_v), V)
    lengths = Int[]
    for ep in 0:n_ep-1
        S, A, rw, term, s_last = rollout(pol, seed * 10_000 + ep)
        push!(lengths, length(rw))
        if method == :reinforce
            w = reward_to_go(rw)
        else
            v = Float64.(vec(V(S))); v_last = term ? 0.0 : Float64(V(s_last)[1])
            if method == :baseline
                G = reward_to_go(rw); w = G .- v; target = Float32.(G)
            else                                                 # :a2c — GAE 이점, critic은 λ-수익을 회귀
                Adv = gae(rw, v, v_last); w = Adv; target = Float32.(Adv .+ v)
            end
            for _ in 1:5
                gv = Flux.gradient(m -> mean(abs2, vec(m(S)) .- target), V)[1]
                ov, V = Optimisers.update!(ov, V, gv)
            end
        end
        gp = Flux.gradient(m -> -logp_sum(m, S, A, w) / 100, pol)[1]
        opi, pol = Optimisers.update!(opi, pol, gp)
    end
    return lengths, pol, V
end

# ---------------------------------------------------------------- 경사 추정량의 분산
function flat_grad(pol, S, A, w)
    θ, re = Flux.destructure(pol)
    return Float64.(only(Zygote.gradient(q -> -logp_sum(re(q), S, A, w), θ)))
end
function variance_experiment(pol; n_fit = 300, n_eval = 300, seed = 123)
    Vf = mlp(1, Xoshiro(seed)); ov = Optimisers.setup(Optimisers.Adam(1e-2), Vf)
    data = [rollout(pol, seed * 100_000 + i) for i in 0:n_fit-1]
    Sx = reduce(hcat, first.(data)); Gy = Float32.(reduce(vcat, [reward_to_go(d[3]) for d in data]))
    for it in 1:1500                                             # 이 고정 정책의 가치 함수를 MC 회귀로 맞춘다
        g = Flux.gradient(m -> mean(abs2, vec(m(Sx)) .- Gy), Vf)[1]
        ov, Vf = Optimisers.update!(ov, Vf, g)
    end
    mean_len = mean(length(d[3]) for d in data)
    eps = [rollout(pol, seed * 100_000 + n_fit + i) for i in 0:n_eval-1]
    bconst = mean(reduce(vcat, [reward_to_go(e[3]) for e in eps]))
    names = ["total return G_0", "gamma^t G_t (exact)", "reward-to-go G_t", "G_t - const b", "G_t - V(s_t)",
             "GAE lambda=0.95", "TD residual (lambda=0)"]
    est = Dict(k => Vector{Vector{Float64}}() for k in names)
    for (S, A, rw, term, s_last) in eps
        G = reward_to_go(rw); v = Float64.(vec(Vf(S))); vl = term ? 0.0 : Float64(Vf(s_last)[1])
        push!(est[names[1]], flat_grad(pol, S, A, fill(G[1], length(rw))))
        push!(est[names[2]], flat_grad(pol, S, A, GAMMA .^ (0:length(rw)-1) .* G))
        push!(est[names[3]], flat_grad(pol, S, A, G))
        push!(est[names[4]], flat_grad(pol, S, A, G .- bconst))
        push!(est[names[5]], flat_grad(pol, S, A, G .- v))
        push!(est[names[6]], flat_grad(pol, S, A, gae(rw, v, vl; lam = 0.95)))
        push!(est[names[7]], flat_grad(pol, S, A, gae(rw, v, vl; lam = 0.0)))
    end
    M(k) = reduce(hcat, est[k])                                  # 매개변수 x 에피소드
    ref, ref0 = vec(mean(M(names[3]); dims = 2)), vec(mean(M(names[2]); dims = 2))
    rows = map(names) do k
        g = M(k); m = vec(mean(g; dims = 2)); tv = sum(var(g; dims = 2))     # 공분산의 대각합(에피소드당 추정량)
        (k, tv, norm(m), sqrt(tv / size(g, 2)), dot(m, ref0) / norm(m) / norm(ref0), dot(m, ref) / norm(m) / norm(ref))
    end
    return mean_len, rows
end

# ---------------------------------------------------------------- 실행
"20 에피소드 이동 평균이 처음 475 이상이 되는 에피소드(창의 끝, 1부터 셈); 없으면 -1"
function first_reach(l; level = 475, w = 20)
    k = findfirst(>=(level), [mean(l[j:j+w-1]) for j in 1:length(l)-w+1])
    return k === nothing ? -1 : k + w - 1
end
for method in (:reinforce, :baseline, :a2c)
    L = [train(method, sd)[1] for sd in 0:4]
    last = [mean(l[end-49:end]) for l in L]
    first475 = first_reach.(L)
    @printf("%-9s: mean length last 50 episodes = %6.1f +- %5.1f (per seed %s); first episode with 20-ep avg >= 475: %s\n",
            method, mean(last), std(last) / sqrt(5), string(round.(Int, last)), string(first475))
end
_, pi_mid, _ = train(:baseline, 0; n_ep = 60)                     # 부분적으로 학습된 고정 정책
mean_len, rows = variance_experiment(pi_mid)
@printf("\nGradient-estimator variance at a fixed policy (mean episode length %.1f), 300 episodes\n", mean_len)
@printf("%-24s %9s %8s %7s %6s %10s %9s\n", "estimator", "trCov", "var/G_t", "|mean|", "s.e.", "cos(exact)", "cos(G_t)")
base = rows[3][2]
for (k, tv, nm, s, c0, c) in rows
    @printf("%-24s %9.3g %8.3f %7.1f %6.1f %10.3f %9.3f\n", k, tv, tv / base, nm, s, c0, c)
end
