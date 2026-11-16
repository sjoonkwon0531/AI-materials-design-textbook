# ch9_common.jl — 9장 Julia 포트의 공용 함수 (신경망 구조에 의존하지 않는 부분)
# 경로, 오차 지표, GELU(erf 정확식)와 그 도함수, PyTorch와 같은 식의 학습률 일정, 평평한 매개변수 벡터 위의
# Adam 한 걸음(Optimisers.jl)과 L-BFGS(Optim.jl, 강한 Wolfe 선 탐색), Crank–Nicolson, 순위 통계.
using LinearAlgebra, Random, Statistics, Printf, Serialization
using SpecialFunctions: erf
import Optimisers, Zygote, Optim, LineSearches

const CH9 = normpath(joinpath(@__DIR__, ".."))
const CH9_DATA = joinpath(CH9, "data")                 # 파이썬 Lab이 만든 npz (FROM_PYTHON = true일 때만 읽는다)
const CH9_OUT = joinpath(@__DIR__, "models")           # Julia 판이 만든 자료와 학습한 매개변수
isdir(CH9_OUT) || mkpath(CH9_OUT)

# ---------------------------------------------------------------- 오차 지표 (배치는 항상 마지막 축)
rel_l2(a, b) = norm(a .- b) / norm(b)
"표본(마지막 축)마다의 상대 L2 오차"
rel_each(p, t) = vec(sqrt.(sum(abs2, p .- t; dims = Tuple(1:ndims(p)-1)) ./ sum(abs2, t; dims = Tuple(1:ndims(t)-1))))
"배치 평균 상대 L2: 연산자 학습의 손실 (PyTorch 판의 rel_l2와 같은 식)"
rel_loss(p, t) = mean(sqrt.(sum(abs2, p .- t; dims = Tuple(1:ndims(p)-1))) ./ sqrt.(sum(abs2, t; dims = Tuple(1:ndims(t)-1))))
"numpy 배열의 축 순서를 뒤집는다: (B, ..., N) → (N, ..., B). 파이썬이 저장한 npz를 column-major 약속으로 읽을 때 쓴다"
revdims(A) = permutedims(A, ndims(A):-1:1)
"사다리꼴 적분"
trapz(y, x) = sum((y[1:end-1] .+ y[2:end]) .* diff(x)) / 2
"가장 짧은 실행 시간(초). 공유 장비에서는 최솟값이 가장 믿을 만하다"
best_time(f; rep = 10) = minimum(@elapsed(f()) for _ in 1:rep)

# ---------------------------------------------------------------- 활성화
"PyTorch nn.GELU의 기본(erf 정확식). NNlib의 gelu는 tanh 근사다"
gelu_exact(x) = x * (1 + erf(x / sqrt(oftype(x, 2)))) / 2
"그 도함수 Φ(x) + x φ(x): 시간 방향 접선을 손으로 전파할 때 쓴다"
dgelu(x) = (1 + erf(x / sqrt(oftype(x, 2)))) / 2 + x * exp(-x^2 / 2) / sqrt(oftype(x, 2pi))

# ---------------------------------------------------------------- 학습률 일정 (s는 0부터 센 갱신 횟수)
"torch CosineAnnealingLR"
cosine_lr(lr0, s, total; eta_min = 0.0) = eta_min + (lr0 - eta_min) * (1 + cos(pi * s / total)) / 2
"torch OneCycleLR(anneal_strategy = cos, div_factor 25, final_div_factor 1e4)의 학습률"
function onecycle_lr(maxlr, s, total; pct = 0.3, div = 25.0, fdiv = 1e4)
    up = pct * total - 1
    lo = maxlr / div
    s <= up && return lo + (maxlr - lo) * (1 - cos(pi * s / up)) / 2
    fin = lo / fdiv
    return fin + (maxlr - fin) * (1 + cos(pi * (s - up) / (total - 1 - up))) / 2
end

# ---------------------------------------------------------------- 평평한 θ 위의 최적화
"""Adam 한 걸음. clip은 전역 기울기 노름 자르기(torch clip_grad_norm_), wd는 torch.optim.Adam(weight_decay)의 결합 L2.
순서는 PyTorch와 같다: 자른 뒤에 감쇠 항을 더한다. (opt, θ)를 돌려준다."""
function adam_step!(opt, θ, g; lr = nothing, wd = 0.0, clip = Inf)
    gn = norm(g)
    gn > clip && (g = g .* eltype(θ)(clip / gn))
    wd > 0 && (g = g .+ eltype(θ)(wd) .* θ)
    lr === nothing || Optimisers.adjust!(opt, lr)
    return Optimisers.update!(opt, θ, g)
end

"loss(θ)를 Adam으로 iters번 줄인다. sched(it)가 있으면 그 학습률을 쓴다. cb(it, θ)는 기록용"
function train_adam(loss, θ; iters, lr = 1e-3, sched = nothing, cb = nothing)
    opt = Optimisers.setup(Optimisers.Adam(lr), θ)
    for it in 0:iters-1
        cb === nothing || cb(it, θ)
        g = Zygote.gradient(loss, θ)[1]
        opt, θ = adam_step!(opt, θ, g; lr = sched === nothing ? nothing : sched(it))
    end
    return θ
end

"""L-BFGS (Optim.jl, 이력 m, 강한 Wolfe 선 탐색). torch.optim.LBFGS(max_iter = iters, history_size = m,
line_search_fn = "strong_wolfe")에 대응한다. 선 탐색의 세부와 멈춤 규칙이 달라 반복 수와 최종 손실은 같지 않다."""
function train_lbfgs(loss, θ; iters, m = 50, gtol = 1e-12)
    iters <= 0 && return θ
    function fg!(F, G, x)
        if G !== nothing
            val, gs = Zygote.withgradient(loss, x)
            G .= gs[1]
            return val
        end
        return loss(x)
    end
    res = Optim.optimize(Optim.only_fg!(fg!), θ, Optim.LBFGS(; m = m, linesearch = LineSearches.StrongWolfe()),
                         Optim.Options(; iterations = iters, g_tol = gtol))
    return Optim.minimizer(res)
end

# ---------------------------------------------------------------- 고전 기준선: Crank–Nicolson
"u_t = D u_xx, u(0, t) = u(1, t) = 0. x와 U((Nx+1) x (Nt+1))를 돌려준다. 삼중대각 계는 LinearAlgebra의 Tridiagonal LU"
function crank_nicolson(D, Nx, Nt, u0; T = 1.0)
    x = collect(range(0, 1; length = Nx + 1)); lam = D * (T / Nt) * Nx^2
    A = lu(Tridiagonal(fill(-lam / 2, Nx - 2), fill(1 + lam, Nx - 1), fill(-lam / 2, Nx - 2)))
    u = u0.(x); u[1] = u[end] = 0.0
    U = zeros(Nx + 1, Nt + 1); U[:, 1] = u
    for n in 1:Nt
        u[2:end-1] = A \ (u[2:end-1] .+ lam / 2 .* (u[3:end] .- 2 .* u[2:end-1] .+ u[1:end-2]))
        U[:, n+1] = u
    end
    return x, U
end

# ---------------------------------------------------------------- 순위 통계
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
"ROC-AUC = Mann–Whitney U / (n₀ n₁)"
function auroc(score, labels::AbstractVector{Bool})
    r = tiedrank(score); n1 = count(labels); n0 = length(labels) - n1
    return (sum(r[labels]) - n1 * (n1 + 1) / 2) / (n0 * n1)
end
"분할 등각 예측의 분위수: ⌈(n + 1)(1 - α)⌉번째로 작은 점수"
conformal_q(s; alpha = 0.1) = sort(s)[ceil(Int, (length(s) + 1) * (1 - alpha))]
