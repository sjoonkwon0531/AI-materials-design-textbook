# lab_9_3.jl — Lab 9.3: 드문 잡음 자료에서 확산계수 추정하기 (labs/92_inverse.py)
# Lab 9.2와 같은 문제에서 D를 모른다. log D를 매개변수 벡터의 마지막 칸에 붙여 신경망과 함께 최적화한다.
#   B1: 흩어진 점 40개, 초기·경계조건은 안다(강제 구속).   B2: 센서 5개 x 시각 16개, 초기·경계조건을 모른다.
# 고전 기준선: B1은 Crank–Nicolson을 순방향 모형으로 쓴 1변수 최소제곱(Brent), B2는 사인 급수 최소제곱(작은 Levenberg–Marquardt).
include("heat_pinn.jl")

const SIGMA, D_TRUE = 0.02, 0.1

function make_data(case, rng)
    if case == :B1
        xd = 0.05 .+ 0.9 .* rand(rng, 40); td = 0.05 .+ 0.95 .* rand(rng, 40)
    else
        xs = [0.15, 0.3, 0.5, 0.7, 0.85]; ts = range(0.1, 1.0; length = 16)
        xd = vec([x for t in ts, x in xs]); td = vec([t for t in ts, x in xs])
    end
    return xd, td, HEAT2.exact.(xd, td, D_TRUE) .+ SIGMA .* randn(rng, length(xd))
end

"PINN 역문제: (D̂, 자료 맞춤 RMSE). 초기 추측 D = 0.5(참값의 5배), 자료 항의 가중치 10"
function run_inverse(case, seed; n_adam = 3000, n_lbfgs = 500)
    xd, td, yd = make_data(case, Xoshiro(seed)); hard = case == :B1
    _, _, θ = init_mlp(HEAT_SIZES; seed); np = length(θ)
    g = Xoshiro(200 + seed)
    xf = hard ? rand(g, 1, 1000) : 0.15 .+ 0.7 .* rand(g, 1, 1000)      # B2: 배치점은 센서가 둘러싼 영역 안에만
    tf = hard ? rand(g, 1, 1000) : 0.1 .+ 0.9 .* rand(g, 1, 1000)
    xr, tr, yr = reshape(xd, 1, :), reshape(td, 1, :), reshape(yd, 1, :)
    loss(p) = mean(abs2, heat_residual(HEAT2, p[1:np], xf, tf, exp(p[np+1]), hard)) +
              10 * mean(abs2, heat_u(HEAT2, p[1:np], xr, tr, hard) .- yr)
    p = vcat(θ, log(0.5))
    p = train_adam(loss, p; iters = n_adam, lr = 2e-3)
    p = train_lbfgs(loss, p; iters = n_lbfgs)
    return exp(p[end]), sqrt(mean(abs2, heat_u(HEAT2, p[1:np], xr, tr, hard) .- yr))
end

"격자 U[i, j] = u(x_i, t_j) 위의 쌍선형 보간 (scipy RegularGridInterpolator의 기본과 같다)"
function bilinear(U, x, t)
    nx, nt = size(U) .- 1
    fx = clamp(x * nx, 0, nx - 1e-9); ft = clamp(t * nt, 0, nt - 1e-9)
    i, j = floor(Int, fx) + 1, floor(Int, ft) + 1; a, b = fx - (i - 1), ft - (j - 1)
    return (1 - a) * (1 - b) * U[i, j] + a * (1 - b) * U[i+1, j] + (1 - a) * b * U[i, j+1] + a * b * U[i+1, j+1]
end

"B1 기준선: D 하나에 대한 최소제곱. 순방향 모형은 CN(100 x 100). (D̂, 순방향 풀이 횟수)"
function classical_B1(seed)
    xd, td, yd = make_data(:B1, Xoshiro(seed)); nsolve = Ref(0)
    function sse(D)
        nsolve[] += 1
        _, U = crank_nicolson(D, 100, 100, HEAT2.u0)
        return sum(abs2, bilinear.(Ref(U), xd, td) .- yd)
    end
    r = Optim.optimize(sse, 0.01, 1.0)                                 # 구간 [0.01, 1] 위의 Brent
    return Optim.minimizer(r), nsolve[]
end

"B2 기준선: 초기조건 없이 D와 사인 모드 4개의 진폭을 맞춘다(동차 Dirichlet 경계조건은 안다고 가정)"
function classical_B2(seed)
    xd, td, yd = make_data(:B2, Xoshiro(seed))
    res(p) = [sum(p[1+k] * exp(-exp(p[1]) * (k * pi)^2 * td[i]) * sin(k * pi * xd[i]) for k in 1:4) - yd[i] for i in eachindex(yd)]
    p = vcat(log(0.5), zeros(4)); r = res(p); cost = sum(abs2, r); mu = 1e-3
    for _ in 1:200
        J = ForwardDiff.jacobian(res, p); A = J' * J; d = diag(A)
        Dm = Diagonal(max.(d, 1e-10 * maximum(d)))                     # Marquardt 척도
        pn, rn, cn = p, r, cost; accepted = false
        while mu < 1e20
            pn = p .- (A + mu * Dm) \ (J' * r); rn = res(pn); cn = sum(abs2, rn)
            if cn < cost
                accepted = true; break                                 # 수락: Gauss–Newton 쪽으로
            end
            mu *= 4                                                    # 거부: 경사 하강 쪽으로
        end
        accepted || break
        done = (cost - cn) < 1e-14 * cost
        p, r, cost, mu = pn, rn, cn, max(mu / 3, 1e-12)
        done && break
    end
    return exp(p[1])
end

function main()
    println("=== inverse problem: identify D from sparse noisy data (noise sigma = 0.02, initial guess D = 0.5) ===")
    for case in (:B1, :B2)
        Ds = Float64[]
        for seed in 0:4
            t0 = time(); Dhat, rm = run_inverse(case, seed)
            push!(Ds, Dhat)
            @printf("  %s seed %d: D_hat = %.4f (true 0.1000), data RMSE %.4f (noise sigma %.2f), %.1f s\n", case, seed, Dhat, rm, SIGMA, time() - t0)
        end
        @printf("  %s PINN : D = %.4f +/- %.4f (5 noise/init seeds), |bias| %.1f %%\n", case, mean(Ds), std(Ds; corrected = false), abs(mean(Ds) - D_TRUE) / D_TRUE * 100)
        if case == :B1
            cl = [classical_B1(s) for s in 0:4]; Dc = first.(cl)
            @printf("  B1 CN + scalar least squares: D = %.4f +/- %.4f (%d forward solves)\n", mean(Dc), std(Dc; corrected = false), cl[1][2])
        else
            Dc = [classical_B2(s) for s in 0:4]
            @printf("  B2 sine-series least squares (BC assumed known): D = %.4f +/- %.4f\n", mean(Dc), std(Dc; corrected = false))
        end
    end
end
main()
