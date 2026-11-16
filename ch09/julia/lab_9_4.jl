# lab_9_4.jl — Lab 9.4: 신경망의 도함수, 자동미분 대 유한차분, 그리고 계수에 따른 비용 (labs/92_ad.py)
# (A) sin 2πx에 맞춘 1-16-16-1 tanh 망의 u_xx를 19개 점에서 구한다. 기준값은 같은 망을 BigFloat(약 40자리)로 계산한 jet.
#     Float64·Float32의 정확한 미분(jet, 중첩 ForwardDiff)과 망을 차분한 값을 비교한다.
# (B) k계 입력 도함수의 비용: 중첩 ForwardDiff(Dual을 k번 겹친다; 비용이 대략 2ᵏ으로 는다) 대
#     Taylor 모드를 손으로 쓴 jet(한 번의 통과로 k계까지; 비용이 k에 대해 다항식으로 는다). TaylorDiff.jl이 같은 일을 자동으로 한다.
include("ch9_common.jl")
include("pinn_common.jl")

"""직선 a + s a1 위에서 4계까지의 도함수 (u, u′, u″, u‴, u⁗). Faà di Bruno 공식에 tanh의 도함수
T′ = 1 - a², T″ = -2a T′, T‴ = T′(6a² - 2), T⁗ = T′(16a - 24a³)을 넣은 것이다."""
function jet4(sizes, θ, a, a1)
    L = length(sizes) - 1; o = 0
    a2 = zero(a1); a3 = zero(a1); a4 = zero(a1)
    for k in 1:L
        n, m = sizes[k], sizes[k+1]
        W = reshape(θ[o+1:o+m*n], m, n); b = θ[o+m*n+1:o+m*n+m]; o += m * n + m
        z = W * a .+ b; z1 = W * a1; z2 = W * a2; z3 = W * a3; z4 = W * a4
        if k < L
            a = tanh.(z); T1 = 1 .- a .^ 2; T2 = -2 .* a .* T1
            T3 = T1 .* (6 .* a .^ 2 .- 2); T4 = T1 .* (16 .* a .- 24 .* a .^ 3)
            a1 = T1 .* z1
            a2 = T2 .* z1 .^ 2 .+ T1 .* z2
            a3 = T3 .* z1 .^ 3 .+ 3 .* T2 .* z1 .* z2 .+ T1 .* z3
            a4 = T4 .* z1 .^ 4 .+ 6 .* T3 .* z1 .^ 2 .* z2 .+ T2 .* (4 .* z1 .* z3 .+ 3 .* z2 .^ 2) .+ T1 .* z4
        else
            a, a1, a2, a3, a4 = z, z1, z2, z3, z4
        end
    end
    return a, a1, a2, a3, a4
end

"k번 겹친 ForwardDiff: f의 k계 도함수 함수"
nested(f, k) = k == 0 ? f : (x -> ForwardDiff.derivative(nested(f, k - 1), x))

function partA()
    SA = [1, 16, 16, 1]
    _, _, θ = init_mlp(SA; seed = 0)
    xs = reshape(collect(range(0, 1; length = 200)), 1, :); ys = sin.(2pi .* xs)
    fitloss(θ) = mean(abs2, mlp_value(SA, θ, xs) .- ys)
    θ = train_adam(fitloss, θ; iters = 1500, lr = 5e-3)               # 망을 자명하지 않게: sin 2πx에 맞춘다
    @printf("A. network 1-16-16-1 tanh fitted to sin(2 pi x): fit RMSE %.1e\n", sqrt(fitloss(θ)))
    xp = collect(range(0.05, 0.95; length = 19))
    setprecision(BigFloat, 140)                                        # 140비트 ≈ 십진 42자리
    ref = Float64.(vec(jet2(SA, big.(θ), reshape(big.(xp), 1, :), [big(1.0)], [big(0.0)])[3]))
    @printf("   reference u_xx by 42-digit arithmetic at 19 points, max|u_xx| = %.2f\n", maximum(abs, ref))
    relerr(v) = maximum(abs, v .- ref) / maximum(abs, ref)
    uxx_jet(T) = Float64.(vec(jet2(SA, T.(θ), reshape(T.(xp), 1, :), T[1], T[0])[3]))
    u1(x) = mlp_value(SA, θ, fill(x, 1, 1))[1]
    function uxx_fd(T, h)                                              # 망을 중심 차분: [u(x+h) - 2u(x) + u(x-h)]/h², 모두 T형으로
        p = T.(θ); xq = T.(xp); hh = T(h)
        u(x) = vec(mlp_value(SA, p, reshape(x, 1, :)))
        return Float64.((u(xq .+ hh) .- 2 .* u(xq) .+ u(xq .- hh)) ./ hh^2)
    end
    @printf("   autodiff (jet)            float64: max rel err %.1e\n", relerr(uxx_jet(Float64)))
    @printf("   autodiff (nested ForwardDiff) f64: max rel err %.1e\n", relerr([nested(u1, 2)(x) for x in xp]))
    @printf("   autodiff (jet)            float32: max rel err %.1e\n", relerr(uxx_jet(Float32)))
    println("   central FD of the network, max rel err:\n        h      float64    float32")
    for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6)
        @printf("     %7.0e  %9.1e  %9.1e\n", h, relerr(uxx_fd(Float64, h)), relerr(uxx_fd(Float32, h)))
    end
end

function partB(; N = 2000)
    SB = [2, 64, 64, 64, 1]
    _, _, θ = init_mlp(SB; seed = 1); rng = Xoshiro(1)
    x, t = rand(rng, N), rand(rng, N); X = permutedims(hcat(x, t))
    u(xi, ti) = mlp_value(SB, θ, reshape([xi, ti], 2, 1))[1]
    kth_nested(k) = [nested(xi -> u(xi, t[i]), k)(x[i]) for i in 1:N]                       # 점마다 Dual을 k번 겹친다
    kth_taylor(k) = vec(k == 1 ? jet1(SB, θ, X, [1.0, 0.0])[2] : k == 2 ? jet2(SB, θ, X, [1.0, 0.0], [0.0, 0.0])[3] : jet4(SB, θ, X, [1.0, 0.0])[k+1])
    # 학습에 필요한 것은 도함수 그 자체가 아니라 '도함수로 만든 손실의 매개변수 기울기'다: jet 위의 Zygote
    grad_taylor(k) = Zygote.gradient(p -> mean(abs2, k == 1 ? jet1(SB, p, X, [1.0, 0.0])[2] : k == 2 ? jet2(SB, p, X, [1.0, 0.0], [0.0, 0.0])[3] : jet4(SB, p, X, [1.0, 0.0])[k+1]), θ)[1]
    t_fwd = best_time(() -> mlp_value(SB, θ, X); rep = 25)
    @printf("B. network 2-64-64-64-1 tanh (float64), N = %d points, minimum of several runs\n   forward pass only: %.2f ms\n", N, 1e3 * t_fwd)
    println("   k   nested ForwardDiff     Taylor jet           (+param grad, Zygote)   max|nested - jet|")
    for k in 1:4
        a = best_time(() -> kth_nested(k); rep = 3); b = best_time(() -> kth_taylor(k); rep = 25); c = best_time(() -> grad_taylor(k); rep = 10)
        @printf("   %d   %8.2f ms (%6.1fx)   %8.2f ms (%5.1fx)   %8.2f ms             %.1e\n",
                k, 1e3 * a, a / t_fwd, 1e3 * b, b / t_fwd, 1e3 * c, maximum(abs, kth_nested(k) .- kth_taylor(k)))
    end
    println("   (the jet for k = 3 computes all four orders; k = 3 and 4 share one pass)")
end

partA(); partB()
