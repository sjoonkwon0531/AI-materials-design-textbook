# lab_9_1.jl — Lab 9.1: 진자의 우변 학습, Neural ODE 대 Hamiltonian NN (labs/91_hnn.py)
# 참 계: H = p²/2 + (1 - cos q), q̇ = p, ṗ = -sin q. 학습은 '이산화한 뒤 최적화': RK4 K걸음을 Zygote로 역전파한다.
# HNN의 벡터장 J∇H_θ는 입력 방향 jet 두 번으로 정확히 구한다(pinn_common.jl). 기준 해와 학습 자료는
# OrdinaryDiffEq.jl(Vern9, 허용오차 1e-11)로 만든다. 연속 수반법으로 학습하려면 SciMLSensitivity.jl·DiffEqFlux.jl이 있다.
include("ch9_common.jl")
include("pinn_common.jl")
using OrdinaryDiffEq
using SpecialFunctions: ellipk

H_true(q, p) = 0.5 * p^2 + (1 - cos(q))
pend!(du, u, _, t) = (du[1] = u[2]; du[2] = -sin(u[1]); nothing)
"기준 해: ts의 각 시각에서 (q, p). 2 x length(ts)"
reference(x0, ts) = reduce(hcat, solve(ODEProblem(pend!, x0, (0.0, ts[end])), Vern9(); reltol = 1e-11, abstol = 1e-12, saveat = ts).u)

const DT, NSNAP, NTRAJ, SIGMA, K = 0.2, 21, 40, 0.02, 5
const S_NODE, S_HNN = [2, 64, 64, 2], [2, 64, 64, 1]
const EQ, EP = [1.0, 0.0], [0.0, 1.0]

"분리선(E = 2) 아래의 궤적 40개, 스냅숏 21개, 잡음 σ = 0.02. 창: 연속한 K + 1개 스냅숏"
function make_data(rng)
    clean = zeros(2, NSNAP, NTRAJ); n = 0; ts = collect(0:NSNAP-1) .* DT
    while n < NTRAJ
        q0, p0 = 4.4 * rand(rng) - 2.2, 3.0 * rand(rng) - 1.5
        0.05 < H_true(q0, p0) < 1.8 || continue
        n += 1; clean[:, :, n] = reference([q0, p0], ts)
    end
    noisy = clean .+ SIGMA .* randn(rng, size(clean)...)
    idx = 1:NSNAP-K
    X0 = reshape(noisy[:, idx, :], 2, :)                                     # 2 x 640
    Y = [reshape(noisy[:, idx .+ j, :], 2, :) for j in 1:K]                  # j걸음 뒤의 스냅숏
    return clean, noisy, X0, Y
end

"Neural ODE: ẋ = f_θ(x), 구조 없음"
node_field(θ, X) = mlp_value(S_NODE, θ, X)
"HNN: ẋ = (∂H_θ/∂p, -∂H_θ/∂q). 벡터장이 만들어질 때부터 심플렉틱 기울기다"
function hnn_field(θ, X)
    dHdq = jet1(S_HNN, θ, X, EQ)[2]; dHdp = jet1(S_HNN, θ, X, EP)[2]
    return vcat(dHdp, -dHdq)
end

function rk4(f, θ, x, h)
    k1 = f(θ, x); k2 = f(θ, x .+ 0.5h .* k1); k3 = f(θ, x .+ 0.5h .* k2); k4 = f(θ, x .+ h .* k3)
    return x .+ h / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"식 (9.1): 적분기를 통과하는 역전파. Adam 1,200회, 학습률 3e-3의 코사인 감쇠"
function fit(field, θ, X0, Y; iters = 1200, lr = 3e-3)
    function loss(θ)
        x = X0; L = 0.0
        for j in 1:K
            x = rk4(field, θ, x, DT)
            L += mean(abs2, x .- Y[j])
        end
        return L / K
    end
    θ = train_adam(loss, θ; iters, lr, sched = it -> cosine_lr(lr, it, iters))
    return θ, loss(θ)
end

function rollout(field, θ, x0, T, h)
    n = round(Int, T / h); out = zeros(2, n + 1); x = reshape(copy(x0), 2, 1); out[:, 1] = x
    for i in 1:n
        x = rk4(field, θ, x, h); out[:, i+1] = x
    end
    return out
end

function main()
    clean, noisy, X0, Y = make_data(Xoshiro(0))
    x0 = [2.0, 0.0]; T = 200.0; h = 0.02; E0 = H_true(x0...)
    tt = collect(0:round(Int, T / h)) .* h
    ref = reference(x0, tt)
    @printf("data: %d trajectories x %d snapshots (dt=%.1f), noise sigma=%.2f; %d windows of K=%d steps\n", NTRAJ, NSNAP, DT, SIGMA, size(X0, 2), K)
    @printf("test orbit: q0=2.0, p0=0, E0=%.4f, period %.3f, T=%.0f\n", E0, 4 * ellipk(sin(1.0)^2), T)
    qq = [q for q in range(-2, 2; length = 41), p in range(-1.5, 1.5; length = 31)]
    pp = [p for q in range(-2, 2; length = 41), p in range(-1.5, 1.5; length = 31)]
    m = H_true.(qq, pp) .< 1.8
    Xg = permutedims(hcat(qq[m], pp[m])); Ft = vcat(pp[m]', -sin.(qq[m])')        # 학습 영역 격자 위의 참 벡터장
    for (name, field, S) in (("NeuralODE-MLP", node_field, S_NODE), ("HNN", hnn_field, S_HNN))
        dE, eT, fe, tr = Float64[], Float64[], Float64[], Float64[]
        for seed in 0:2
            _, _, θ = init_mlp(S; seed)
            seed == 0 && name == "HNN" && println("  jet vs nested ForwardDiff (1st, 2nd derivative): ", check_jets(S, θ))
            t0 = time(); θ, L = fit(field, θ, X0, Y); w = time() - t0
            traj = rollout(field, θ, x0, T, h); E = H_true.(traj[1, :], traj[2, :])
            push!(fe, sqrt(mean(sum(abs2, field(θ, Xg) .- Ft; dims = 1)))); push!(tr, w)
            push!(dE, E[end] - E0); push!(eT, maximum(abs, E .- E0))
            @printf("  %s seed %d: train loss %.2e, field RMSE %.4f, |x-x_ref|(t=20) %.3f, E(T)-E0 %+.4f, max|E-E0| %.4f, train %.0f s",
                    name, seed, L, fe[end], norm(traj[:, 1001] .- ref[:, 1001]), dE[end], eT[end], w)
            if name == "HNN"                                                        # H_θ 자체는 적분 오차 수준으로 보존된다
                Hth = vec(mlp_value(S_HNN, θ, traj)); @printf(", max|H_theta drift| %.1e", maximum(abs, Hth .- Hth[1]))
            end
            println()
        end
        @printf("  %s: E(T)-E0 = %+.4f +/- %.4f; max|E-E0| = %.4f +/- %.4f; field RMSE %.4f +/- %.4f; train %.0f s\n",
                name, mean(dE), std(dE; corrected = false), mean(eT), std(eT; corrected = false), mean(fe), std(fe; corrected = false), mean(tr))
    end
    true_field(_, x) = vcat(x[2:2, :], -sin.(x[1:1, :]))                            # 같은 걸음의 RK4를 참 벡터장에: 적분 오차만
    E = H_true.(eachrow(rollout(true_field, nothing, x0, T, h))...)
    @printf("  true field + RK4 (h=%.2f): max|E-E0| = %.1e  (integrator error is negligible)\n", h, maximum(abs, E .- E0))
end
main()
