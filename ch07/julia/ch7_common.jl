# ch7_common.jl — 7장 Julia 포트의 공용 함수 (신경망 프레임워크에 의존하지 않는 부분)
# 데이터 읽기(파이썬이 한 번 내보낸 CSV), 분할 인덱스, 전처리(VarianceThreshold + StandardScaler + clip),
# 지표, 학습률 일정(cosine, one-cycle, warmup+cosine), PyTorch와 같은 식의 AdamW/Adam+L2 규칙.
# Flux를 쓰는 Lab은 flux_tools.jl, Lux를 쓰는 Lab은 mlip_common.jl을 이 파일 다음에 include한다
# (Flux와 Lux는 Dense, Chain 같은 이름을 함께 내보내므로 한 파일에서 둘 다 using하지 않는다).
using CSV, DataFrames, Statistics, LinearAlgebra, Random, Printf
import Optimisers

const CH7 = normpath(joinpath(@__DIR__, ".."))
const DATA = joinpath(CH7, "data")
const LABS = joinpath(CH7, "labs")

# ---------------------------------------------------------------- 데이터
"Magpie 132 특징 + expt_gap. X는 n x p(행 = 관측), 파이썬 DataFrame과 같은 배치"
function load_magpie()
    df = CSV.read(joinpath(DATA, "magpie_expt_gap.csv"), DataFrame)
    feat = [c for c in names(df) if c ∉ ("formula", "target")]
    return Matrix{Float64}(df[:, feat]), Vector{Float64}(df.target), String.(df.formula)
end

"6장 Lab 6.7의 정제된 특징 표(149개)"
function load_features149()
    df = CSV.read(joinpath(DATA, "67_features_clean.csv"), DataFrame)
    feat = [c for c in names(df) if c ∉ ("formula", "target", "chemsys", "reduced")]
    return Matrix{Float64}(df[:, feat]), Vector{Float64}(df.target), String.(df.chemsys)
end

"파이썬이 내보낸 0-based 인덱스(data/splits/*.csv, 열 이름 idx)를 1-based로 읽는다"
read_idx(name) = CSV.read(joinpath(DATA, "splits", name), DataFrame).idx .+ 1

"화학식의 원소 기호 집합 (괄호·계수는 무시; 'N'과 'Na', 'Nb'를 구분한다)"
elements_of(formula) = Set(m.match for m in eachmatch(r"[A-Z][a-z]?", formula))

# ---------------------------------------------------------------- 전처리
"VarianceThreshold(thr) + StandardScaler(ddof = 0) + clip. 학습 부분으로만 맞춘다"
struct Prep
    keep::Vector{Int}
    mu::Vector{Float64}
    sd::Vector{Float64}
    clipv::Float64
end
function fit_prep(X; thr = 1e-6, clipv = 5.0)
    v = vec(var(X; dims = 1, corrected = false))
    keep = thr === nothing ? collect(1:size(X, 2)) : findall(>(thr), v)
    mu = vec(mean(X[:, keep]; dims = 1))
    sd = vec(std(X[:, keep]; dims = 1, corrected = false))
    sd[sd .== 0] .= 1.0                                        # sklearn: 분산 0인 열은 스케일 1
    return Prep(keep, mu, sd, clipv)
end
(p::Prep)(X) = clamp.((X[:, p.keep] .- p.mu') ./ p.sd', -p.clipv, p.clipv)
"신경망 입력: 특징 x 관측(열 = 관측), Float32"
feats(p::Prep, X) = Float32.(permutedims(p(X)))

# ---------------------------------------------------------------- 지표
mae(a, b) = mean(abs.(a .- b))
rmse(a, b) = sqrt(mean(abs2.(a .- b)))

# ---------------------------------------------------------------- 학습률 일정 (s는 0부터 센 갱신 횟수)
"torch CosineAnnealingLR"
cosine_lr(lr0, s, total; eta_min = 0.0) = eta_min + (lr0 - eta_min) * (1 + cos(pi * s / total)) / 2

"torch OneCycleLR(anneal_strategy = cos, div_factor 25, final_div_factor 1e4)의 학습률. 모멘텀(β₁) 순환은 생략"
function onecycle_lr(maxlr, s, total; pct = 0.3, div = 25.0, fdiv = 1e4)
    up = pct * total - 1
    lo = maxlr / div
    s <= up && return lo + (maxlr - lo) * (1 - cos(pi * s / up)) / 2
    fin = lo / fdiv
    return fin + (maxlr - fin) * (1 + cos(pi * (s - up) / (total - 1 - up))) / 2
end

"Lab 7.3 Part C: 5% 선형 warmup 뒤 cosine (배율; 최대 학습률에 곱한다)"
warm_cos(s, total) = min(1.0, (s + 1) / (0.05 * total)) * 0.5 * (1 + cos(pi * min(1.0, s / total)))

# ---------------------------------------------------------------- PyTorch와 같은 식의 Adam 계열
# Optimisers.jl의 AdamW는 판에 따라 감쇠 항에 학습률을 곱하는지가 달라서(0.3 대 0.4), 식을 직접 적었다.
#   couple_l2 = false: torch.optim.AdamW   θ ← θ - η λ θ - η m̂ / (√v̂ + ϵ)       (감쇠가 정규화 밖에 분리)
#   couple_l2 = true : torch.optim.Adam(weight_decay = λ)   g ← g + λ θ 다음 Adam  (L2가 정규화 안에 결합)
struct TorchAdamW <: Optimisers.AbstractRule
    eta::Float64
    beta::Tuple{Float64,Float64}
    lambda::Float64
    epsilon::Float64
    couple_l2::Bool
end
TorchAdamW(eta = 1e-3; beta = (0.9, 0.999), lambda = 1e-2, epsilon = 1e-8, couple_l2 = false) =
    TorchAdamW(eta, beta, lambda, epsilon, couple_l2)
AdamL2(eta, lambda) = TorchAdamW(eta; lambda = lambda, couple_l2 = true)

Optimisers.init(o::TorchAdamW, x::AbstractArray) = (zero(x), zero(x), o.beta)
function Optimisers.apply!(o::TorchAdamW, state, x::AbstractArray{T}, dx) where {T}
    eta, lam, eps = T(o.eta), T(o.lambda), T(o.epsilon)
    b1, b2 = T(o.beta[1]), T(o.beta[2])
    mt, vt, bt = state
    g = o.couple_l2 ? dx .+ lam .* x : dx
    @. mt = b1 * mt + (1 - b1) * g
    @. vt = b2 * vt + (1 - b2) * g^2
    c1, c2 = 1 - T(bt[1]), 1 - T(bt[2])
    step = @. eta * (mt / c1) / (sqrt(vt / c2) + eps)
    dxnew = o.couple_l2 ? step : step .+ eta .* lam .* x
    return (mt, vt, bt .* o.beta), dxnew
end
