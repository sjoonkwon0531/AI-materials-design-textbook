# ch8_common.jl — 8장 Julia 포트의 공용 함수 (신경망 프레임워크에 의존하지 않는 부분)
# 경로, 작은 수치 도구(softmax, 무작위 동점 깨기, 누적 최댓값, 선형 보간), 벡터 위의 Adam(numpy 판의 손 Adam과 같은 식),
# 통계 도구(표준오차, ROC-AUC, Spearman). 7장의 데이터를 쓰는 Lab(8.9–8.11)은 7장의 ch7_common.jl도 include한다.
using Random, Statistics, LinearAlgebra, Printf

const CH8 = normpath(joinpath(@__DIR__, ".."))
const CH8_DATA = joinpath(CH8, "data")
const CH8_LABS = joinpath(CH8, "labs")
const CH7_JL = normpath(joinpath(CH8, "..", "ch07", "julia"))     # 7장 Julia 포트(ch7_common.jl, flux_tools.jl, mlip_common.jl)

# ---------------------------------------------------------------- 작은 수치 도구
softmax_vec(z) = (e = exp.(z .- maximum(z)); e ./ sum(e))
"최댓값이 여럿이면 그중 하나를 균일하게 고른다 (numpy의 rng.choice(np.flatnonzero(q == q.max())))"
argmax_tie(rng, q) = (m = maximum(q); rand(rng, findall(==(m), q)))
cummax(v) = accumulate(max, v)
"np.interp(x, xp, fp): xp는 증가, 범위 밖은 끝값"
function interp1(x::Real, xp::AbstractVector, fp::AbstractVector)
    x <= xp[1] && return fp[1]
    x >= xp[end] && return fp[end]
    k = searchsortedlast(xp, x)
    xp[k+1] == xp[k] && return fp[k+1]
    return fp[k] + (fp[k+1] - fp[k]) * (x - xp[k]) / (xp[k+1] - xp[k])
end
se(x) = std(x) / sqrt(length(x))
"파이썬이 내보낸 0-based 인덱스(data/splits/*.csv, 머리줄 idx)를 1-based로 읽는다 (CSV.jl 없이)"
read_idx8(name) = parse.(Int, readlines(joinpath(CH8_DATA, "splits", name))[2:end]) .+ 1

# ---------------------------------------------------------------- 벡터 위의 Adam (numpy 판 Lab 8.12의 손 Adam과 같은 식)
mutable struct VAdam
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
    b1::Float64
    b2::Float64
    eps::Float64
end
VAdam(n; b1 = 0.9, b2 = 0.999, eps = 1e-8) = VAdam(zeros(n), zeros(n), 0, b1, b2, eps)
"상승(ascent) 방향의 한 걸음: θ ← θ + lr m̂ / (√v̂ + ϵ)"
function adam_ascent!(o::VAdam, θ, grad, lr)
    o.t += 1
    @. o.m = o.b1 * o.m + (1 - o.b1) * grad
    @. o.v = o.b2 * o.v + (1 - o.b2) * grad^2
    @. θ += lr * (o.m / (1 - o.b1^o.t)) / (sqrt(o.v / (1 - o.b2^o.t)) + o.eps)
    return θ
end

# ---------------------------------------------------------------- 통계
"ROC-AUC = Mann–Whitney U / (n₀ n₁), 동점은 평균 순위"
function auroc(labels::AbstractVector{Bool}, score::AbstractVector)
    r = tiedrank(score)
    n1 = count(labels); n0 = length(labels) - n1
    return (sum(r[labels]) - n1 * (n1 + 1) / 2) / (n0 * n1)
end
"평균 순위(동점 처리)"
function tiedrank(x)
    p = sortperm(x); r = zeros(length(x)); i = 1
    while i <= length(x)
        j = i
        while j < length(x) && x[p[j+1]] == x[p[i]]
            j += 1
        end
        r[p[i:j]] .= (i + j) / 2
        i = j + 1
    end
    return r
end
spearman(a, b) = cor(tiedrank(a), tiedrank(b))
