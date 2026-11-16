# lab_9_6.jl — Lab 9.6: 두 척도 Poisson 문제, 스펙트럼 편향, Fourier 특성, NTK 불균형, 그리고 최적화기 (labs/92_spectral.py)
# -u″ = f on (0, 1), u(0) = u(1) = 0, 정확한 해 u = sin 2πx + 0.1 sin 50πx.
#   A. PDE 없이 u를 회귀: 어느 주파수를 먼저 배우는가.   C. 초기화에서의 경험적 NTK 블록 K_uu(경계), K_rr(잔차)의 대각합.
#   B. PINN: tanh MLP 대 Fourier 특성 망(σ = 10); Adam(같은 가중치 / NTK 대각합 가중치), Adam + L-BFGS, 핵(NTK) 꼴의 Levenberg–Marquardt.
# Fourier 특성 z = [cos 2πBx; sin 2πBx]는 x의 함수이므로 jet2의 입력에 z′, z″을 해석적으로 넣는다.
# 야코비안 ∂(잔차)/∂θ는 Zygote.jacobian으로 구한다(출력 수만큼의 역전파; 매개변수 1.7만 개에는 전방 모드보다 싸다).
include("ch9_common.jl")
include("pinn_common.jl")

u_ex(x) = sin(2pi * x) + 0.1 * sin(50pi * x)
f_rhs(x) = (2pi)^2 * sin(2pi * x) + 0.1 * (50pi)^2 * sin(50pi * x)
const NR = 256
const XR = reshape(collect(range(0, 1; length = NR + 2))[2:end-1], 1, :)       # 안쪽 배치점
const XB = [0.0 1.0]
const XE = reshape(collect(range(0, 1; length = 2001)), 1, :)

"B가 비어 있으면 보통 MLP(입력 x), 아니면 고정된 무작위 주파수 B ~ N(0, σ²)의 Fourier 특성 망"
struct SNet
    sizes::Vector{Int}
    B::Vector{Float64}
end
function make_net(seed; sigma = nothing, m = 64, width = 64)
    B = sigma === nothing ? Float64[] : sigma .* randn(Xoshiro(1000 + seed), m)
    sizes = [(sigma === nothing ? 1 : 2m), width, width, width, 1]
    return SNet(sizes, B), init_mlp(sizes; seed)[3]
end
"입력 jet (z, z′, z″): 점 x (1 x N)에서"
function features(net::SNet, x)
    isempty(net.B) && return x, [1.0], [0.0]
    w = 2pi .* net.B; ph = w .* x                                              # m x N
    return vcat(cos.(ph), sin.(ph)), vcat(-w .* sin.(ph), w .* cos.(ph)), vcat(-w .^ 2 .* cos.(ph), -w .^ 2 .* sin.(ph))
end
const FEATS = IdDict{SNet,Any}()                                               # 특성은 상수이므로 망마다 한 번만 만든다
feats(net) = get!(() -> (r = features(net, XR), b = features(net, XB), e = features(net, XE), f = f_rhs.(XR)), FEATS, net)

value(net, θ, z) = mlp_value(net.sizes, θ, z[1])
resid(net, θ, F) = -jet2(net.sizes, θ, F.r...)[3] .- F.f                       # -u″ - f (1 x NR)
rel_err(net, θ) = rel_l2(value(net, θ, feats(net).e), u_ex.(XE))
"예측의 사인 급수 계수: k = 2 (목표 1), k = 50 (목표 0.1)"
function mode_amps(net, θ)
    up = vec(value(net, θ, feats(net).e)); x = vec(XE)
    return 2 * trapz(up .* sin.(2pi .* x), x), 2 * trapz(up .* sin.(50pi .* x), x)
end

"경험적 NTK K = J Jᵀ의 블록: (Tr K_uu, Tr K_rr[, λ_max(K_uu), K_rr의 고윳값])"
function ntk_traces(net, θ; full = false)
    F = feats(net)
    Ju = Zygote.jacobian(p -> vec(value(net, p, F.b)), θ)[1]                   # 2 x P
    Jr = Zygote.jacobian(p -> vec(resid(net, p, F)), θ)[1]                     # NR x P
    full || return sum(abs2, Ju), sum(abs2, Jr)
    return sum(abs2, Ju), sum(abs2, Jr), eigmax(Symmetric(Ju * Ju')), eigvals(Symmetric(Jr * Jr'))
end

"Adam(지수 감쇠: 1,000회마다 x0.9), 필요하면 뒤에 L-BFGS. weighting = :ntk이면 1,000회마다 NTK 대각합 가중치를 다시 잰다"
function train_pinn(net, θ, weighting, iters; n_lbfgs = 0, lr = 1e-3)
    F = feats(net); lam_b, lam_r = 1.0, 1.0
    opt = Optimisers.setup(Optimisers.Adam(lr), θ)
    for it in 0:iters-1
        if weighting == :ntk && it % 1000 == 0
            tu, tr = ntk_traces(net, θ); lam_b, lam_r = (tu + tr) / tu, (tu + tr) / tr
        end
        lb, lr_w = lam_b, lam_r                                                # 닫힘이 잡을 값
        g = Zygote.gradient(p -> lr_w * mean(abs2, resid(net, p, F)) + lb * mean(abs2, value(net, p, F.b)), θ)[1]
        opt, θ = adam_step!(opt, θ, g; lr = lr * 0.9^(it / 1000))
    end
    θ = train_lbfgs(p -> lam_r * mean(abs2, resid(net, p, F)) + lam_b * mean(abs2, value(net, p, F.b)), θ; iters = n_lbfgs, m = 100, gtol = 1e-14)
    return θ, lam_b / lam_r
end

"핵 꼴의 Levenberg–Marquardt: dθ = -Jᵀ (K + μI)⁻¹ r, K = J Jᵀ (경험적 NTK). 풀어야 할 계의 크기는 (NR + 2)다"
function train_lm(net, θ, iters; mu = 1e-2)
    F = feats(net)
    resvec(p) = vcat(vec(resid(net, p, F)) ./ sqrt(NR), vec(value(net, p, F.b)) ./ sqrt(2))
    r = resvec(θ); L = sum(abs2, r)
    for _ in 1:iters
        J = Zygote.jacobian(resvec, θ)[1]; Km = J * J'
        while mu < 1e12
            dθ = -(J' * ((Km + mu * I) \ r)); rn = resvec(θ .+ dθ); Ln = sum(abs2, rn)
            if Ln < L
                θ, r, L, mu = θ .+ dθ, rn, Ln, max(mu / 3, 1e-12); break
            end
            mu *= 4
        end
    end
    return θ, 1.0
end

function main(; NAD = 3000)
    println("=== A. regression of u(x) (no PDE): amplitude of the k=2 and k=50 sine modes (targets 1 and 0.1) ===")
    xt = reshape(collect(range(0, 1; length = 512)), 1, :); yt = u_ex.(xt)
    for (name, sigma) in (("tanh MLP", nothing), ("Fourier features sigma=10", 10.0))
        net, θ = make_net(0; sigma); zt = features(net, xt)[1]; rec = String[]
        function cb(it, p)
            it in (100, 300, 1000, 2000) || return
            a2, a50 = mode_amps(net, p); push!(rec, @sprintf("it %4d: a2 %.3f a50 %+.4f", it, a2, a50))
        end
        θ = train_adam(p -> mean(abs2, mlp_value(net.sizes, p, zt) .- yt), θ; iters = 5000, lr = 1e-3, cb)
        a2, a50 = mode_amps(net, θ); push!(rec, @sprintf("it 5000: a2 %.3f a50 %+.4f", a2, a50))
        @printf("  %-26s %s\n  %-26s final rel L2 error %.2e\n", name, join(rec, " | "), "", rel_err(net, θ))
    end
    println("=== C. empirical NTK at initialization (256 residual points, 2 boundary points) ===")
    for (name, sigma) in (("tanh MLP", nothing), ("FF sigma=1", 1.0), ("FF sigma=10", 10.0), ("FF sigma=50", 50.0))
        tu, tr, _, er = ntk_traces(make_net(0; sigma)...; full = true)
        @printf("  %-12s: Tr K_uu/N_b = %9.3e   Tr K_rr/N_r = %9.3e   ratio = %8.2e   lam_max(K_rr) = %.2e\n", name, tu / 2, tr / NR, tr / NR / (tu / 2), er[end])
    end
    println("  linear random-Fourier-feature model, Eq. (9.7): ratio = mean (2 pi b)^4 -> 3 (2 pi sigma)^4")
    for sigma in (1.0, 10.0, 50.0)
        b = sigma .* randn(Xoshiro(0), 4096)
        @printf("    sigma = %4.0f: mean (2 pi b)^4 = %.3e,  3 (2 pi sigma)^4 = %.3e\n", sigma, mean((2pi .* b) .^ 4), 3 * (2pi * sigma)^4)
    end
    @printf("=== B. PINN, N_r = %d; rel L2 error on 2001 points ===\n", NR)
    runs = [("tanh MLP, Adam", nothing, :adam, :equal), ("FF sigma=10, Adam, equal weights", 10.0, :adam, :equal),
            ("FF sigma=10, Adam, NTK weights", 10.0, :adam, :ntk), ("FF sigma=10, Adam + L-BFGS", 10.0, :lbfgs, :equal),
            ("FF sigma=10, Levenberg-Marquardt", 10.0, :lm, :equal), ("tanh MLP, Levenberg-Marquardt", nothing, :lm, :equal)]
    for (name, sigma, method, wt) in runs
        net, θ = make_net(0; sigma); t0 = time()
        θ, ratio = method == :adam ? train_pinn(net, θ, wt, NAD) :
                   method == :lbfgs ? train_pinn(net, θ, wt, NAD ÷ 6; n_lbfgs = NAD - NAD ÷ 6) : train_lm(net, θ, 60)
        a2, a50 = mode_amps(net, θ); ub = value(net, θ, feats(net).b)
        @printf("  %-33s: rel L2 = %.2e | a2 %.3f a50 %.4f u(0) %+.1e u(1) %+.1e L_r %.1e lam_b/lam_r %.1e, %.0f s\n",
                name, rel_err(net, θ), a2, a50, ub[1], ub[2], mean(abs2, resid(net, θ, feats(net))), ratio, time() - t0)
    end
    println("=== baseline: second-order finite differences (tridiagonal solve) ===")
    for n in (256, 1024, 4096)
        x = collect(range(0, 1; length = n + 1)); h = 1 / n; u = zeros(n + 1)
        solve_fd() = SymTridiagonal(fill(2.0, n - 1), fill(-1.0, n - 2)) \ (h^2 .* f_rhs.(x[2:end-1]))
        u[2:end-1] = solve_fd()
        @printf("  N = %5d: rel L2 error %.2e, %.3f ms\n", n, rel_l2(u, u_ex.(x)), 1e3 * best_time(solve_fd; rep = 20))
    end
end
main()
