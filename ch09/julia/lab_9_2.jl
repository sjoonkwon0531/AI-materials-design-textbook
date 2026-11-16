# lab_9_2.jl — Lab 9.2: 1차원 확산 방정식의 PINN, 벌점 구속 대 강제 구속, 그리고 Crank–Nicolson (labs/92_heat.py)
# u_t = D u_xx, D = 0.1, u(x, 0) = sin πx + 0.5 sin 3πx. 신경망 2-32-32-32-1 tanh, 배치점 1,000개,
# Adam 3,000회(학습률 1e-3) 뒤 L-BFGS 최대 1,500회. 잔차의 u_t, u_xx는 pinn_common.jl의 jet으로 정확히 구한다.
include("heat_pinn.jl")

function main()
    pb = HEAT2
    println("=== Part A: forward problem, soft vs hard constraints (2-32-32-32-1 tanh, 1000 collocation points) ===")
    for hard in (false, true)
        rel, mx, wall = Float64[], Float64[], Float64[]
        for seed in 0:2
            model, st, θ = init_mlp(HEAT_SIZES; seed)
            if seed == 0 && !hard                                 # 같은 망을 Lux로 평가한 값, 그리고 중첩 ForwardDiff와의 일치
                X = rand(Xoshiro(7), 2, 5); c1, c2 = check_jets(HEAT_SIZES, θ)
                @printf("  Lux forward vs flat-θ forward: %.1e; jet vs nested ForwardDiff: %.1e, %.1e\n",
                        maximum(abs, lux_value(model, st, HEAT_SIZES, θ, X) .- mlp_value(HEAT_SIZES, θ, X)), c1, c2)
            end
            loss = heat_loss(pb, heat_points(Xoshiro(100 + seed)), hard)
            t0 = time()
            θ = train_adam(loss, θ; iters = 3000, lr = 1e-3); e_adam = heat_errors(pb, θ, hard)[1]
            θ = train_lbfgs(loss, θ; iters = 1500)
            w = time() - t0
            r, m, _ = heat_errors(pb, θ, hard)
            push!(rel, r); push!(mx, m); push!(wall, w)
            @printf("  %s seed %d: rel L2 after Adam %.2e -> after L-BFGS %.2e, max err %.2e, %.1f s\n", hard ? "hard" : "soft", seed, e_adam, r, m, w)
        end
        @printf("  %s: rel L2 = %.2e +/- %.1e, max = %.2e, %.1f s (3 seeds)\n", hard ? "hard" : "soft", mean(rel), std(rel; corrected = false), mean(mx), mean(wall))
    end
    println("=== Crank-Nicolson baseline (tridiagonal LU) ===")
    for n in (25, 50, 100, 200, 400)
        r, m, _, dt = cn_error(pb, n, n)
        @printf("  Nx=Nt=%4d: rel L2 %.2e, max err %.2e, %.2f ms\n", n, r, m, 1e3 * dt)
    end
end
main()
