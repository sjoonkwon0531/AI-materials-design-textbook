# lab_9_7.jl — Lab 9.7: 이류 방정식의 실패 모드와 세 가지 처방 (labs/92_convection.py)
# u_t + β u_x = 0, x ∈ [0, 2π), t ∈ [0, 1], u(x, 0) = sin x, 주기 경계. 정확한 해 u = sin(x - βt).
# 신경망 2-50-50-50-50-1 tanh(입력을 [-1, 1]로 정규화), 손실 = 잔차 + 초기조건 + 주기 경계조건(모두 벌점, 가중치 1).
# 1계 도함수만 필요하므로 jet1 두 번(ξ 방향, τ 방향)이면 된다. 연쇄 법칙: u_x = u_ξ/π, u_t = u_τ · 2/(t1 - t0).
# 줄인 포트: 기본 PINN은 β마다 시드 하나, 처방은 '단계마다 L-BFGS로 끝내는 curriculum'과 '시간 행진', 주기 임베딩을 시드 하나로 돌린다.
include("ch9_common.jl")
include("pinn_common.jl")

const SV, SP = [2, 50, 50, 50, 50, 1], [3, 50, 50, 50, 50, 1]
const E1, E2, E3 = [1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]
struct CNet
    periodic::Bool                                           # true면 입력이 (cos x, sin x, t): x 방향 주기성이 정확히 성립
    t0::Float64
    t1::Float64
end
CNet(; periodic = false, t0 = 0.0, t1 = 1.0) = CNet(periodic, t0, t1)
inputs(net::CNet, x, t) = net.periodic ? vcat(cos.(x), sin.(x), t) : vcat(x ./ pi .- 1, 2 .* (t .- net.t0) ./ (net.t1 - net.t0) .- 1)
sizes(net::CNet) = net.periodic ? SP : SV
u_value(net, θ, x, t) = mlp_value(sizes(net), θ, inputs(net, x, t))
"u_t + β u_x (1 x N)"
function residual(net, θ, x, t, beta)
    X = inputs(net, x, t)
    if net.periodic
        ux = jet1(SP, θ, X, vcat(-sin.(x), cos.(x), zero(x)))[2]; ut = jet1(SP, θ, X, E3)[2]
    else
        ux = jet1(SV, θ, X, E1)[2] ./ pi; ut = jet1(SV, θ, X, E2)[2] .* (2 / (net.t1 - net.t0))
    end
    return ut .+ beta .* ux
end

function points(seed; nf = 1000, ni = 200, nb = 100, t0 = 0.0, t1 = 1.0)
    g = Xoshiro(500 + seed)
    return 2pi .* rand(g, 1, nf), t0 .+ (t1 - t0) .* rand(g, 1, nf), 2pi .* rand(g, 1, ni), t0 .+ (t1 - t0) .* rand(g, 1, nb)
end
"손실의 세 항 (잔차, 초기조건, 주기 경계조건). ic는 초기선 점 xi에서의 목표값(1 x ni)"
function loss_parts(net, θ, beta, pts, ic)
    xf, tf, xi, tb = pts
    Lr = mean(abs2, residual(net, θ, xf, tf, beta))
    Li = mean(abs2, u_value(net, θ, xi, fill(net.t0, size(xi))) .- ic)
    Lb = net.periodic ? 0.0 : mean(abs2, u_value(net, θ, zero(tb), tb) .- u_value(net, θ, fill(2pi, size(tb)), tb))
    return Lr, Li, Lb
end
function optimize(net, θ, beta, pts, ic, n_adam, n_lbfgs)
    function loss(p)
        Lr, Li, Lb = loss_parts(net, p, beta, pts, ic)
        return Lr + Li + Lb
    end
    θ = train_adam(loss, θ; iters = n_adam, lr = 1e-3)
    return train_lbfgs(loss, θ; iters = n_lbfgs)
end

const XG = collect(0:255) .* (2pi / 256); const TG = collect(range(0, 1; length = 101))
"256 x 101 격자 위의 예측"
predict(net, θ; ts = TG) = reshape(u_value(net, θ, reshape([x for x in XG, t in ts], 1, :), reshape([t for x in XG, t in ts], 1, :)), 256, :)
exact(beta; ts = TG) = [sin(x - beta * t) for x in XG, t in ts]
err_t(U, beta) = [norm(U[:, j] .- sin.(XG .- beta * TG[j])) / norm(sin.(XG .- beta * TG[j])) for j in eachindex(TG)]

"고전 기준선: Lax–Wendroff (CFL 0.8), t = 1에서의 오차와 걸음 수"
function lax_wendroff(beta, N; cfl = 0.8)
    x = collect(0:N-1) .* (2pi / N); dx = 2pi / N
    nt = ceil(Int, beta / (cfl * dx)); c = beta * (1 / nt) / dx
    u = sin.(x)
    for _ in 1:nt
        up, um = circshift(u, -1), circshift(u, 1)
        u = u .- 0.5c .* (up .- um) .+ 0.5c^2 .* (up .- 2 .* u .+ um)
    end
    return rel_l2(u, sin.(x .- beta)), nt
end

function main(; NA = 2400, NL = 600)
    @printf("=== A. vanilla PINN vs beta (%d Adam + up to %d L-BFGS iterations, 1000 collocation points) ===\n", NA, NL)
    for beta in (1, 10, 20, 30)
        net = CNet(); pts = points(0); θ = init_mlp(SV; seed = 0)[3]; t0 = time()
        θ = optimize(net, θ, beta, pts, sin.(pts[3]), NA, NL)
        U = predict(net, θ); et = err_t(U, beta); Lr, Li, Lb = loss_parts(net, θ, beta, pts, sin.(pts[3]))
        @printf("  beta = %2d: rel L2 = %.3f | L_r %.1e L_ic %.1e L_bc %.1e, err(t=0.1) %.2f, err(t=1) %.2f, mean|u(t=1)| %.3f, %.0f s\n",
                beta, rel_l2(U, exact(beta)), Lr, Li, Lb, et[11], et[end], mean(abs, U[:, end]), time() - t0)
        if beta == 30                                        # 진단: 학습 점 위의 잔차 대 새 점 위의 잔차, 파의 진폭
            g = Xoshiro(999); xt, tt = 2pi .* rand(g, 1, 20000), rand(g, 1, 20000)
            amp = sqrt.(2 .* vec(mean(abs2, U; dims = 1)))
            @printf("    residual MSE on the 1000 training points %.2e; on 20000 fresh points %.2e; amplitude at t = 0, 0.5, 1: %.3f, %.3f, %.3f\n",
                    Lr, mean(abs2, residual(net, θ, xt, tt, beta)), amp[1], amp[51], amp[101])
        end
    end
    beta = 30
    println("=== B. remedies at beta = 30: every variant uses 3000 loss evaluations in total, like the vanilla runs ===")
    # (1) curriculum: β를 5에서 30까지 올리고, 단계마다 L-BFGS로 끝낸다
    net = CNet(); pts = points(0); θ = init_mlp(SV; seed = 0)[3]
    for b in (5, 10, 15, 20, 25, 30)
        θ = optimize(net, θ, b, pts, sin.(pts[3]), 150, 350)
    end
    U = predict(net, θ); et = err_t(U, beta)
    @printf("  curriculum 6 stages x (150 Adam + 350 L-BFGS): rel L2 = %.4f | err(t=0.1) %.3f, err(t=1) %.3f\n", rel_l2(U, exact(beta)), et[11], et[end])
    # (2) 시간 행진: 구간 6개. 앞 구간의 예측이 다음 구간의 초기조건이고, 가중치도 이어받는다(구간마다 시간 좌표를 [-1, 1]로)
    nwin = 6; U = zeros(256, 101); θ = init_mlp(SV; seed = 0)[3]; prev = nothing
    for k in 0:nwin-1
        ta, tb = k / nwin, (k + 1) / nwin; net = CNet(; t0 = ta, t1 = tb); pts = points(10k; t0 = ta, t1 = tb)
        ic = prev === nothing ? sin.(pts[3]) : u_value(prev[1], prev[2], pts[3], fill(ta, size(pts[3])))
        θ = optimize(net, copy(θ), beta, pts, ic, 150, 350)
        cols = findall(t -> ta - 1e-12 <= t <= tb + 1e-12, TG)
        U[:, cols] = predict(net, θ; ts = TG[cols]); prev = (net, θ)
    end
    et = err_t(U, beta)
    @printf("  time marching, 6 slabs x (150 Adam + 350 L-BFGS): rel L2 = %.4f | err(t=0.1) %.3f, err(t=1) %.3f\n", rel_l2(U, exact(beta)), et[11], et[end])
    # (3) 주기 임베딩 (강제 주기 경계조건), 나머지는 기본 학습 그대로
    net = CNet(; periodic = true); pts = points(0); θ = init_mlp(SP; seed = 0)[3]
    θ = optimize(net, θ, beta, pts, sin.(pts[3]), NA, NL)
    U = predict(net, θ); et = err_t(U, beta)
    @printf("  periodic embedding (cos x, sin x, t), vanilla training: rel L2 = %.4f | err(t=0.1) %.3f, err(t=1) %.3f\n", rel_l2(U, exact(beta)), et[11], et[end])
    println("=== C. classical baseline at beta = 30: Lax-Wendroff (CFL 0.8), error at t = 1 ===")
    for N in (128, 512, 2048)
        e, nt = lax_wendroff(30, N)
        @printf("  N = %4d, %5d steps: rel L2 error %.2e, %.1f ms\n", N, nt, e, 1e3 * best_time(() -> lax_wendroff(30, N); rep = 3))
    end
end
main()
