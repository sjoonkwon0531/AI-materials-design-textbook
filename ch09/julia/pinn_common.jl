# pinn_common.jl — PINN Lab(9.1–9.7, 9.15, 9.18)의 공용 부분: Lux.jl로 정의한 tanh MLP와, 입력 도함수를 층마다 전파하는 jet
#
# PINN의 손실은 신경망의 입력 도함수(u_t, u_x, u_xx)를 담고, 학습은 그 손실을 다시 매개변수로 미분한다.
# PyTorch는 역방향 미분을 겹쳐 쓴다(create_graph = True). Julia에서는 "역방향(Zygote) 위의 전방(입력)" 조합을 쓴다.
#   · 입력 방향: 값과 방향 도함수의 묶음 (a, a′, a″)을 층마다 연쇄 법칙으로 전파한다(전방 모드, Taylor 모드를 손으로 쓴 것).
#     선형층은 z = W a + b, z′ = W a′, z″ = W a″,  tanh 층은 a = tanh z, a′ = (1 - a²) z′, a″ = (1 - a²) z″ - 2a (1 - a²) z′².
#   · 매개변수 방향: 위의 전파는 행렬곱과 broadcast뿐인 평범한 배열 코드이므로 Zygote가 그대로 역전파한다.
# 차분은 어디에도 없고, tanh가 매끄러우므로 모든 도함수가 기계 정밀도까지 정확하다(check_jets가 중첩 ForwardDiff와 맞춰 본다).
# 매개변수는 평평한 Float64 벡터 θ 하나다. 순서는 Lux의 layer_k.weight(out x in, column-major), layer_k.bias다.
using Lux, Zygote, ForwardDiff, Random, LinearAlgebra

# ---------------------------------------------------------------- Lux 모형과 초기화
"PyTorch nn.Linear의 기본 초기화: 가중치와 편향 모두 U(±1/√fan_in)"
torch_uniform(nin) = (rng, dims...) -> (rand(rng, Float64, dims...) .* 2 .- 1) ./ sqrt(nin)
"Glorot 정규 초기화 (DeepXDE의 \"Glorot normal\"; 편향은 0)"
glorot_normal64(rng, dims...) = randn(rng, Float64, dims...) .* sqrt(2 / (dims[1] + dims[2]))
zeros64(rng, dims...) = zeros(Float64, dims...)
function pinn_dense(nin, nout, act, init)
    init == :glorot && return Dense(nin => nout, act; init_weight = glorot_normal64, init_bias = zeros64)
    return Dense(nin => nout, act; init_weight = torch_uniform(nin), init_bias = torch_uniform(nin))
end
"sizes = [입력, 은닉..., 출력]인 tanh MLP (마지막 층은 선형)"
tanh_mlp(sizes; init = :torch) =
    Chain([pinn_dense(sizes[k], sizes[k+1], k < length(sizes) - 1 ? tanh : identity, init) for k in 1:length(sizes)-1]...)

"Lux 모형, 상태, 평평한 초기 매개변수 θ (Float64)"
function init_mlp(sizes; seed = 0, init = :torch)
    model = tanh_mlp(sizes; init)
    ps, st = Lux.setup(Xoshiro(seed), model)
    θ = reduce(vcat, [vcat(vec(Float64.(l.weight)), vec(Float64.(l.bias))) for l in values(ps)])
    return model, st, θ
end
n_params(sizes) = sum(sizes[k+1] * (sizes[k] + 1) for k in 1:length(sizes)-1)

"평평한 θ → Lux 매개변수 NamedTuple (layer_k = (weight, bias)). AD 밖에서만 쓴다"
function lux_params(sizes, θ)
    L = length(sizes) - 1; o = 0; layers = []
    for k in 1:L
        n, m = sizes[k], sizes[k+1]
        push!(layers, (weight = reshape(θ[o+1:o+m*n], m, n), bias = θ[o+m*n+1:o+m*n+m])); o += m * n + m
    end
    return NamedTuple{Tuple(Symbol(:layer_, k) for k in 1:L)}(Tuple(layers))
end
"Lux 순전파 (X는 d x N). mlp_value와 같은 값이어야 한다"
lux_value(model, st, sizes, θ, X) = first(model(X, lux_params(sizes, θ), st))

# ---------------------------------------------------------------- 값과 jet
"값만: a (d x N) → (출력 x N)"
function mlp_value(sizes, θ, a)
    L = length(sizes) - 1; o = 0
    for k in 1:L
        n, m = sizes[k], sizes[k+1]
        W = reshape(θ[o+1:o+m*n], m, n); b = θ[o+m*n+1:o+m*n+m]; o += m * n + m
        z = W * a .+ b
        a = k < L ? tanh.(z) : z
    end
    return a
end

"""1차 jet: 입력 a(d x N)와 그 방향 도함수 a1 → (u, u′).
a1은 d x N 행렬이거나, 모든 점에서 같은 방향이면 길이 d의 벡터다(예: x 방향 [1, 0], t 방향 [0, 1])."""
function jet1(sizes, θ, a, a1)
    L = length(sizes) - 1; o = 0
    for k in 1:L
        n, m = sizes[k], sizes[k+1]
        W = reshape(θ[o+1:o+m*n], m, n); b = θ[o+m*n+1:o+m*n+m]; o += m * n + m
        z = W * a .+ b; z1 = W * a1
        if k < L
            a = tanh.(z); a1 = (1 .- a .^ 2) .* z1
        else
            a, a1 = z, z1
        end
    end
    return a, a1
end

"""2차 jet: (a, a′, a″) → (u, u′, u″). 좌표 입력이면 a″ = 0이고, 입력이 좌표의 함수(주기 임베딩, Fourier 특성)이면
그 함수의 1·2차 도함수를 a1, a2로 넣는다."""
function jet2(sizes, θ, a, a1, a2)
    L = length(sizes) - 1; o = 0
    for k in 1:L
        n, m = sizes[k], sizes[k+1]
        W = reshape(θ[o+1:o+m*n], m, n); b = θ[o+m*n+1:o+m*n+m]; o += m * n + m
        z = W * a .+ b; z1 = W * a1; z2 = W * a2
        if k < L
            a = tanh.(z); d = 1 .- a .^ 2
            a1 = d .* z1
            a2 = d .* z2 .- 2 .* a .* d .* z1 .^ 2
        else
            a, a1, a2 = z, z1, z2
        end
    end
    return a, a1, a2
end

"""jet을 '입력 위의 중첩 ForwardDiff'와 맞춰 본다: 무작위 점 x, 무작위 방향 v에서 s ↦ u(x + s v)의 1·2차 도함수.
두 값의 차이(절댓값)를 돌려준다. 기계 정밀도 수준이어야 한다."""
function check_jets(sizes, θ; rng = Xoshiro(1))
    x = rand(rng, sizes[1]); v = randn(rng, sizes[1])
    f(s) = mlp_value(sizes, θ, reshape(x .+ s .* v, :, 1))[1]
    d1 = ForwardDiff.derivative(f, 0.0)
    d2 = ForwardDiff.derivative(s -> ForwardDiff.derivative(f, s), 0.0)
    _, j1, j2 = jet2(sizes, θ, reshape(x, :, 1), v, zero(v))
    return abs(j1[1] - d1), abs(j2[1] - d2)
end
