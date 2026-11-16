# sam.jl — SAM, ASAM, F-SAM과 날카로움 진단 (labs/83_sam.py의 Julia 대응)
# 매개변수를 평평한 벡터 θ 하나로 다룬다. Flux 모형은 Flux.destructure로 (θ, re)를 얻고, Lux 매개변수는 ComponentArray가
# 그대로 θ다. SAM 한 걸음은 Optimisers.jl 상태(opt) 위의 '두 번 계산하는 갱신'이다.
#   1) g = ∇L(θ)                                      (첫 번째 순전파·역전파)
#   2) ε = ρ g/‖g‖ (SAM),  ρ T²g/‖Tg‖, T = |θ| + η (ASAM),  ρ d/‖d‖, d = g - σ m (F-SAM; m은 g의 EMA)
#   3) g̃ = ∇L(θ + ε)                                  (두 번째 순전파·역전파)
#   4) Optimisers.update!(opt, θ, g̃)                  (기반 최적화기는 θ에서 g̃로 한 걸음; 감쇠·모멘트도 θ 기준)
# torch 판(83_sam.py)과 같은 식이며, ε 분모의 1e-12도 같다.
using Optimisers, Zygote, ForwardDiff, LinearAlgebra, Random, Statistics

"F-SAM의 상태: 전체 기울기의 지수 이동 평균 m (첫 단계 전에는 없음)"
mutable struct SAMState
    m::Any
end
SAMState() = SAMState(nothing)

"SAM 계열의 섭동 ε (θ와 같은 모양). F-SAM이면 ss.m을 갱신한다"
function sam_perturbation(θ, g, ss::SAMState; rho, mode = :sam, eta = 0.01, sigma = 1.0, lmbda = 0.6)
    T0 = eltype(θ); rho, eta, sigma, lmbda = T0(rho), T0(eta), T0(sigma), T0(lmbda)
    d = g
    if mode == :fsam
        if ss.m === nothing
            ss.m = copy(g)                                   # 첫 단계: d = g, m = g
        else
            d = g .- sigma .* ss.m                           # 미니배치 기울기에서 전체 기울기 성분을 뺀다
            ss.m .= lmbda .* ss.m .+ (1 - lmbda) .* g
        end
    end
    if mode == :asam
        Tw = abs.(θ) .+ eta
        return rho .* Tw .^ 2 .* d ./ (norm(Tw .* d) + T0(1e-12))
    end
    return rho .* d ./ (norm(d) + T0(1e-12))
end

"""SAM 계열 한 걸음. lossθ(θ)는 이번 미니배치의 손실. mode = :none(기반 최적화기 한 걸음만), :sam, :asam, :fsam.
반환 (opt, θ, θ에서의 손실). 기울기 계산이 두 번이라 비용은 보통 걸음의 약 두 배다."""
function sam_update!(opt, θ, lossθ, ss::SAMState = SAMState(); rho = 0.05, mode = :sam, kw...)
    L, gs = Zygote.withgradient(lossθ, θ); g = gs[1]
    if mode != :none
        e = sam_perturbation(θ, g, ss; rho, mode, kw...)
        g = only(Zygote.gradient(lossθ, θ .+ e))            # w + ε에서의 기울기
    end
    opt, θ = Optimisers.update!(opt, θ, g)                  # w로 돌아와 기반 최적화기 한 걸음
    return opt, θ, L
end

# ---------------------------------------------------------------- 날카로움 진단
"""정확한 Hessian–벡터 곱 Hv = d/dt ∇L(θ + t v) |_{t=0} (forward-over-reverse: Zygote 기울기를 ForwardDiff의 Dual로 미분).
PyTorch의 이중 역전파 HVP와 같은 양이다. ReLU의 꺾이는 점에서도 두 번째 도함수를 0으로 두므로 차분처럼 가짜 곡률이 생기지 않는다.
lossθ는 Dual 원소의 θ를 받아야 한다(Flux 모형이면 아래 chain_mse를 쓴다)."""
hvp_exact(lossθ, θ, v) = ForwardDiff.derivative(t -> only(Zygote.gradient(lossθ, θ .+ t .* v)), zero(eltype(θ)))

"""중심 차분 HVP [∇L(θ + h v) - ∇L(θ - h v)]/(2h) (θ는 Float64; 상대 오차 ~h²). 손실이 매끄러울 때만 쓴다.
ReLU 망에서는 θ ± h v가 꺾이는 점을 가로지르면 기울기의 점프가 1/(2h)배로 들어가 λ_max가 틀린다(Lab 8.11).
Lab 8.10의 MLIP(shifted softplus, 매끄러운 cosine 절단)처럼 손실에 이미 중첩 AD가 들어 있을 때의 대안이다."""
function hvp_fd(lossθ, θ, v; h = 1e-5)
    gp = only(Zygote.gradient(lossθ, θ .+ h .* v))
    gm = only(Zygote.gradient(lossθ, θ .- h .* v))
    return (gp .- gm) ./ (2h)
end

"""Dense 층과 활성화 함수만으로 된 Flux Chain(7장 mlp_chain)의 평균 제곱 오차를 (평평한 Float64 θ, 손실 L(θ))로 만든다.
Flux.destructure의 재구성 함수 re는 원래 원소형으로 사영하므로 ForwardDiff의 Dual을 통과시키지 못한다. 그래서 L은 θ를 직접
잘라 층마다 W(column-major)와 b를 꺼내 계산하며, 어떤 원소형의 θ도 받는다. 편향이 있는 Dense를 가정한다."""
function chain_mse(model, X, y)
    layers = model.layers
    θ = reduce(vcat, [vcat(vec(Float64.(l.weight)), Float64.(l.bias)) for l in layers if !(l isa Function)])
    X64, y64 = Float64.(X), Float64.(vec(y))
    function L(q)
        h = X64; o = 0
        for l in layers
            if l isa Function                                   # 활성화 층(relu, swish)
                h = l.(h)
            else                                                # Dense: W는 (출력, 입력)
                m, n = size(l.weight)
                W = reshape(q[o+1:o+m*n], m, n); b = q[o+m*n+1:o+m*n+m]; o += m * n + m
                h = l.σ.(W * h .+ b)
            end
        end
        return mean(abs2, vec(h) .- y64)
    end
    return θ, L
end

"""최대 Hessian 고윳값: 거듭제곱 반복과 Rayleigh 몫 vᵀHv (‖v‖ = 1, Ch.1 §1.5.2). 반환 (λ, 이력).
기본 HVP는 정확한 forward-over-reverse(hvp_exact)다. 거듭제곱 반복은 절댓값이 가장 큰 고윳값으로 |λ₂/λ₁|ᵏ의 속도로
수렴하므로, 수렴값이 필요하면 iters를 크게, tol = 0으로 둔다(파이썬 판은 같은 HVP 위의 Lanczos, scipy eigsh)."""
function lambda_max(lossθ, θ; iters = 40, seed = 0, tol = 1e-4, hvp = hvp_exact)
    v = copy(θ); v .= randn(Xoshiro(seed), length(θ)); v ./= norm(v)
    lam = 0.0; hist = Float64[]
    for k in 1:iters
        Hv = hvp(lossθ, θ, v)
        lam_new = dot(v, Hv); push!(hist, lam_new)
        v = Hv ./ norm(Hv)
        if k > 3 && abs(lam_new - lam) < tol * abs(lam_new)
            return lam_new, hist
        end
        lam = lam_new
    end
    return lam, hist
end
"""척도 정규화 날카로움 λ_max(T H T), T = |θ₀|: 좌표 u = θ/|θ₀|에서의 Hessian (u₀ = 1).
θ₀의 0인 성분은 이 좌표에서 움직이지 않는다(torch 판과 같음)."""
function lambda_max_adaptive(lossθ, θ0; kw...)
    w0 = copy(θ0)
    return lambda_max(u -> lossθ(u .* w0), one.(θ0); kw...)
end
