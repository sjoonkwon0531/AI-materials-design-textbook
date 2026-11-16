# lab_9_15.jl — Lab 9.15: 1D 열방정식 PINN, 벌점 구속, 강제 구속, Crank–Nicolson (labs/95_dde_heat.py)
# 파이썬 판은 DeepXDE(PyTorch 백엔드)를 쓴다. Julia에서 그 자리에 있는 것은 NeuralPDE.jl(ModelingToolkit.jl의 기호로 PDE와
# 경계조건을 적고 PhysicsInformedNN으로 이산화한 뒤 Optimization.jl로 푼다)이다. 이 책에서는 그 API를 실행으로 확인하지
# 못했으므로, 같은 설정을 heat_pinn.jl의 처음부터 쓴 PINN으로 옮겼다:
#   u_t = D u_xx, D = 0.1, u(x, 0) = sin πx (Ch.1 연습문제 1.8.D의 문제), FNN 2-32-32-32-1 tanh, Glorot 정규 초기화,
#   영역 안 1,000점(+ 경계 200 + 초기 200), Adam 3,000회(학습률 1e-3) 뒤 L-BFGS(이력 50, 최대 2,000회).
# DeepXDE의 Hammersley 표본 대신 균등 난수를 쓰고, L-BFGS는 scipy가 아니라 Optim.jl의 것이다.
include("heat_pinn.jl")

function main()
    pb = HEAT1
    println("Crank-Nicolson (nx = nt = n):   n   max|err| at t=1   rel. L2 on its own grid   time (ms)")
    for n in (25, 50, 100, 200, 400, 800)
        r, _, e1, dt = cn_error(pb, n, n)
        @printf("                              %4d   %.2e          %.2e                %8.2f\n", n, e1, r, 1e3 * dt)
    end
    for hard in (false, true)
        _, _, θ = init_mlp(HEAT_SIZES; seed = 0, init = :glorot)
        pts = heat_points(Xoshiro(0)); loss = heat_loss(pb, pts, hard)
        @printf("\nPINN, %s: FNN 2-32-32-32-1 tanh, %d training points\n", hard ? "hard constraints (output transform)" : "soft constraints (PDE + BC + IC losses)", hard ? 1000 : 1400)
        println("  stage        time (s)   loss        rel. L2 (101x101)   max|err|")
        t0 = time()
        function record(tag, p)
            r, m, _ = heat_errors(pb, p, hard)
            @printf("  %-11s  %7.1f   %.1e      %.2e            %.2e\n", tag, time() - t0, loss(p), r, m)
        end
        θ = train_adam(loss, θ; iters = 3000, lr = 1e-3, cb = (it, p) -> it == 1000 && record("Adam 1000", p))
        record("Adam 3000", θ)
        θ = train_lbfgs(loss, θ; iters = 2000)
        record("+ L-BFGS", θ)
        U = heat_errors(pb, θ, hard)[3]; xg = range(0, 1; length = 101)
        @printf("  u(0.5, 0.5): PINN %.6f, exact %.6f;  u(0.5, 1): PINN %.6f, exact %.6f\n", U[51, 51], exp(-0.1 * pi^2 * 0.5), U[51, 101], exp(-0.1 * pi^2))
        @printf("  max|err| on the line t = 1: %.2e\n", maximum(abs, U[:, 101] .- exp(-0.1 * pi^2) .* sin.(pi .* xg)))
    end
end
main()
