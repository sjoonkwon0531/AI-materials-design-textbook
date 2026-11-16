# lab_7_2.jl — Lab 7.2: 손으로 쓴 역전파, 그 비용, 깊은 망의 기울기 흐름 (labs/71_backprop.py)
# Part A: 식 (7.2)의 행렬 역전파(Float64)를 Zygote와 중심 차분에 맞춘다. 배치는 파이썬과 같이 (특징, 배치).
# Part B: 순전파 대 순전파+역전파 시간(Float32, BLAS 스레드 1). Part C: 깊이 30, 폭 64 MLP의 층별 ||dL/dW_l||.
include("ch7_common.jl")
include("flux_tools.jl")
using LinearAlgebra

silu(z) = z / (1 + exp(-z))
dsilu(z) = (s = 1 / (1 + exp(-z)); s * (1 + z * (1 - s)))

"Glorot normal 가중치와 N(0, 0.1²) 편향 (파이썬과 같은 분포; 난수열은 다르다)"
function init(rng, sizes)
    Ws = [randn(rng, sizes[k+1], sizes[k]) .* sqrt(2 / (sizes[k] + sizes[k+1])) for k in 1:length(sizes)-1]
    bs = [0.1 .* randn(rng, sizes[k+1]) for k in 1:length(sizes)-1]
    return Ws, bs
end

"순전파: 사전활성 z_l과 활성 a_l(a[1] = X)을 모두 저장한다. 마지막 층은 선형"
function forward(Ws, bs, X)
    a = [X]; z = Matrix{Float64}[]
    for l in eachindex(Ws)
        zl = Ws[l] * a[end] .+ bs[l]
        push!(z, zl); push!(a, l == length(Ws) ? zl : silu.(zl))
    end
    return z, a
end

"L = (1/2B) Σ (ŷ - y)²의 역전파: δ_L = (ŷ - y)/B, dL/dW_l = δ_l a_{l-1}ᵀ, δ_{l-1} = (W_lᵀ δ_l) ⊙ f'(z_{l-1})"
function backward(Ws, z, a, Y)
    B = size(Y, 2)
    delta = (a[end] - Y) ./ B
    gW = similar(Ws); gb = Vector{Vector{Float64}}(undef, length(Ws))
    for l in length(Ws):-1:1
        gW[l] = delta * a[l]'
        gb[l] = vec(sum(delta; dims = 2))
        l > 1 && (delta = (Ws[l]' * delta) .* dsilu.(z[l-1]))
    end
    return gW, gb
end

"같은 망을 배열 변경 없이 (Zygote는 push! 같은 변경을 미분하지 못한다)"
function mlp_out(Ws, bs, X)
    h = X
    for l in eachindex(Ws)
        h = Ws[l] * h .+ bs[l]
        l < length(Ws) && (h = silu.(h))
    end
    return h
end
loss_of(Ws, bs, X, Y) = 0.5 * mean(abs2, mlp_out(Ws, bs, X) .- Y)

function partA()
    rng = Xoshiro(0)
    sizes = [8, 32, 16, 1]; B = 64
    Ws, bs = init(rng, sizes)
    X = randn(rng, sizes[1], B); Y = randn(rng, 1, B)
    z, a = forward(Ws, bs, X)
    gW, gb = backward(Ws, z, a, Y)
    # 자동미분: 같은 함수를 Zygote로 (Float64)
    gWz, gbz = Zygote.gradient((W, b) -> loss_of(W, b, X, Y), Ws, bs)
    dmax = maximum(max(maximum(abs.(gW[l] - gWz[l])), maximum(abs.(gb[l] - gbz[l]))) for l in eachindex(Ws))
    println("=== Part A: hand-written backprop vs Zygote (Float64) ===")
    @printf("loss %.12f; max |grad_hand - grad_Zygote| = %.2e (largest entry %.3f)\n",
            loss_of(Ws, bs, X, Y), dmax, maximum(maximum(abs, g) for g in gW))
    eps = 1e-6; i, j = 4, 6                                   # 파이썬의 [3, 5]
    Wp = deepcopy(Ws); Wp[1][i, j] += eps
    Wm = deepcopy(Ws); Wm[1][i, j] -= eps
    fd = (loss_of(Wp, bs, X, Y) - loss_of(Wm, bs, X, Y)) / (2eps)
    @printf("finite difference dL/dW1[%d,%d] = %.10f   backprop = %.10f   |diff| = %.1e\n", i, j, fd, gW[1][i, j], abs(fd - gW[1][i, j]))
end

function partB()
    BLAS.set_num_threads(1)
    println("=== Part B: forward vs forward+backward (Float32, 1 BLAS thread, best of 30) ===")
    for width in (256, 1024)
        rng = Xoshiro(1)
        net = Chain([tdense(rng, width, width, swish) for _ in 1:4]..., tdense(rng, width, 1))
        xb = randn(rng, Float32, width, 512)
        fwd() = sum(net(xb))
        fwdbwd() = Flux.gradient(m -> sum(m(xb)), net)
        fwd(); fwdbwd()                                      # JIT 예열
        best(f) = minimum(@elapsed(f()) for _ in 1:30)
        tf, tfb = best(fwd), best(fwdbwd)
        @printf("width %5d: forward %7.2f ms   forward+backward %7.2f ms   ratio %4.2f\n", width, tf * 1e3, tfb * 1e3, tfb / tf)
    end
end

"깊이 D, 폭 W 망. init ∈ (:xavier, :he, :small, :large), 잔차와 LayerNorm은 선택"
struct Deep{L,N,O,F}
    lins::L
    norms::N
    out::O
    act::F
    residual::Bool
end
Flux.@layer Deep trainable = (lins, norms, out)
function Deep(rng, act, init; D = 30, W = 64, residual = false, norm = false)
    winit = Dict(:xavier => Flux.glorot_normal(rng), :he => Flux.kaiming_normal(rng; gain = sqrt(2.0f0)),
                 :small => (dims...) -> 0.05f0 .* randn(rng, Float32, dims...),
                 :large => (dims...) -> 0.3f0 .* randn(rng, Float32, dims...))[init]
    lins = [Dense(W => W; init = winit, bias = true) for _ in 1:D]   # Flux의 편향 기본값은 0 (파이썬 zeros_와 같다)
    norms = [norm ? LayerNorm(W) : identity for _ in 1:D]
    return Deep(lins, norms, tdense(rng, W, 1), act, residual)
end
function (m::Deep)(x)
    for (lin, nm) in zip(m.lins, m.norms)
        hnew = m.act.(lin(nm(x)))
        x = m.residual ? x .+ hnew : hnew
    end
    return m.out(x)
end

function partC()
    println("=== Part C: per-layer ||dL/dW_l|| in a depth-30, width-64 MLP ===")
    rng = Xoshiro(0)
    xc = randn(rng, Float32, 64, 256); yc = randn(rng, Float32, 1, 256)
    configs = [("sigmoid, Xavier", sigmoid, :xavier, false, false), ("tanh, Xavier", tanh, :xavier, false, false),
               ("ReLU, N(0,0.05^2)", relu, :small, false, false), ("ReLU, N(0,0.3^2)", relu, :large, false, false),
               ("ReLU, He", relu, :he, false, false), ("ReLU, He + residual + LayerNorm", relu, :he, true, true)]
    for (name, act, ini, resid, nm) in configs
        net = Deep(Xoshiro(1), act, ini; residual = resid, norm = nm)
        g = only(Flux.gradient(m -> mean(abs2, m(xc) .- yc), net))
        G = [norm(g.lins[l].weight) for l in 1:30]
        @printf("%-34s layer1 %9.2e  layer15 %9.2e  layer30 %9.2e  ratio first/last %9.2e\n", name, G[1], G[15], G[30], G[1] / G[30])
    end
end

partA(); partB(); partC()
