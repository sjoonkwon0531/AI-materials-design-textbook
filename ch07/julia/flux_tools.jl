# flux_tools.jl — Flux.jl로 쓰는 Lab(7.2, 7.3, 7.5–7.7, 7.17, 7.18, 7.20)의 공용 도구
# 배치 약속: Flux의 Dense는 (특징 x 배치), Conv는 (길이[, 높이], 채널, 배치). 가중치 W는 (출력, 입력)으로
# PyTorch의 nn.Linear.weight와 같은 모양이다.
using Flux, Optimisers, Zygote, MLUtils, NNlib, Random, Statistics

"PyTorch nn.Linear 기본 초기화와 같은 분포: W, b ~ U(-1/√fan_in, 1/√fan_in)"
function tdense(rng::AbstractRNG, nin::Int, nout::Int, act = identity; bias = true)
    k = 1.0f0 / sqrt(Float32(nin))
    W = (rand(rng, Float32, nout, nin) .* 2 .- 1) .* k
    b = bias ? (rand(rng, Float32, nout) .* 2 .- 1) .* k : false
    return Dense(W, b, act)
end

"Linear와 활성화를 따로 둔 MLP (PyTorch의 nn.Sequential과 같은 층 번호; LoRA 등이 Linear만 감쌀 수 있다)"
function mlp_chain(rng, sizes::Vector{Int}, act = swish)
    layers = Any[]
    for k in 1:length(sizes)-1
        push!(layers, tdense(rng, sizes[k], sizes[k+1]))
        k < length(sizes) - 1 && push!(layers, act)
    end
    return Chain(layers...)
end

"파라미터 수"
nparams(m) = sum(length, Flux.trainables(m))

"(입력, 목표)를 무작위 순서의 미니배치로 (마지막 조각 포함; torch randperm 루프와 같은 모양)"
batches(rng, n, bs) = (p = randperm(rng, n); [p[i:min(i + bs - 1, n)] for i in 1:bs:n])

"MAE (예측을 0에서 자른다; 밴드갭은 음수가 될 수 없다)"
gap_mae(m, X, y) = mean(abs.(max.(vec(m(X)), 0.0f0) .- y))
