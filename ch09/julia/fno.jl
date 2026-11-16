# fno.jl — 처음부터 쓴 Fourier Neural Operator (1차원·2차원)와 학습 루프 (labs/93_models.py, labs/95_fno2d.py)
# Lux와 같은 '명시적 매개변수' 방식이다: 매개변수는 NamedTuple이고 모형은 그것을 받는 순수 함수다.
# 학습할 때는 Optimisers.destructure로 평평한 Float32 벡터 θ와 재구성 함수 re를 얻어 Zygote로 미분한다.
# 배치 약속(column-major, numpy 축을 뒤집은 것):
#   1차원: (채널, N, 배치)             — PyTorch (배치, 채널, N)
#   2차원: (채널, N2, N1, 배치)        — PyTorch 채널-마지막 (배치, N1, N2, 채널)의 축을 뒤집은 것
# Fourier 층 (K v)(x) = F⁻¹[R(k) · F v](x)의 복소 가중치 R = Rr + i·Ri는 실수 배열 둘로 저장한다. 복소 행렬곱은
#   (Rr + i Ri)(xr + i xi) = (Rr xr - Ri xi) + i (Rr xi + Ri xr)
# 로 풀어 실수 batched_mul 네 번으로 계산한다(모드마다 행렬 하나). PyTorch의 Adam도 복소 매개변수를 실수부·허수부
# 두 실수로 다루므로 같은 최적화다. rfft/irfft의 역전파 규칙은 AbstractFFTs.jl이 준다.
# ch9_common.jl(gelu_exact, dgelu, adam_step!, onecycle_lr)을 먼저 include한다.
using FFTW, NNlib, Zygote, Random, Statistics
import Optimisers, ChainRulesCore, Functors

# ---------------------------------------------------------------- 점별 선형층
"U(±1/√fan_in): PyTorch nn.Linear와 nn.Conv1d(·, ·, 1)의 기본 초기화"
uinit(rng, fan_in, dims...) = (rand(rng, Float32, dims...) .* 2.0f0 .- 1.0f0) ./ sqrt(Float32(fan_in))
linear(rng, nin, nout) = (W = uinit(rng, nin, nout, nin), b = uinit(rng, nin, nout))
"점별 선형층(1 x 1 합성곱): v (C, 공간..., B) → (C′, 공간..., B)"
pw(l, v) = reshape(l.W * reshape(v, size(v, 1), :), size(l.W, 1), size(v)[2:end]...) .+ l.b
"편향 없는 판: 접선(도함수)을 전파할 때"
pw0(l, v) = reshape(l.W * reshape(v, size(v, 1), :), size(l.W, 1), size(v)[2:end]...)
"실수 매개변수의 수 (복소 가중치는 둘로 센다)"
count_params(p) = length(Optimisers.destructure(p)[1])

# ---------------------------------------------------------------- FNO-1D
"Fourier 층: 가중치 (C′, C, 남길 모드 수), 모드 0..kmax만 남기고 나머지는 0. 어떤 N에서도 같은 가중치를 쓴다"
function spectral1d(l, v)
    C, N, B = size(v); nf = N ÷ 2 + 1
    m = min(size(l.Rr, 3), nf)
    vh = permutedims(rfft(v, 2)[:, 1:m, :], (1, 3, 2))                  # (C, B, m): 마지막 축이 모드
    xr, xi = real(vh), imag(vh)
    Rr, Ri = l.Rr[:, :, 1:m], l.Ri[:, :, 1:m]
    yr = batched_mul(Rr, xr) .- batched_mul(Ri, xi)
    yi = batched_mul(Rr, xi) .+ batched_mul(Ri, xr)
    y = permutedims(complex.(yr, yi), (1, 3, 2))                        # (C′, m, B)
    return irfft(cat(y, zeros(eltype(y), size(y, 1), nf - m, B); dims = 2), N, 2)
end

"PyTorch FNO1d와 같은 분포의 초기화: R = s · 복소 정규(실수부·허수부 분산 1/2), s = 1/width²"
function init_fno1d(rng; in_ch = 1, out_ch = 1, width = 32, kmax = 16, depth = 4)
    s = 1.0f0 / (width * width) / sqrt(2.0f0)
    layer() = (K = (Rr = s .* randn(rng, Float32, width, width, kmax + 1), Ri = s .* randn(rng, Float32, width, width, kmax + 1)),
               W = linear(rng, width, width))
    return (lift = linear(rng, in_ch, width), layers = Tuple(layer() for _ in 1:depth),
            q1 = linear(rng, width, 2width), q2 = linear(rng, 2width, out_ch))
end

"좌표 채널 x/N (평행이동 등변성을 깬다). 미분 대상이 아니다"
gridchannel(T, N, B) = repeat(reshape(T.((0:N-1) ./ N), 1, N, 1), 1, 1, B)
ChainRulesCore.@non_differentiable gridchannel(T::Any, N::Any, B::Any)

"u = Q ∘ (W_L + K_L) ∘ σ(W_{L-1} + K_{L-1}) ∘ … ∘ P (a). 마지막 Fourier 층 뒤에는 활성화가 없다"
function fno1d_net(p, a; grid = false)
    v = pw(p.lift, grid ? vcat(a, gridchannel(eltype(a), size(a, 2), size(a, 3))) : a)
    nl = length(p.layers)
    for i in 1:nl
        l = p.layers[i]
        v = spectral1d(l.K, v) .+ pw(l.W, v)
        if i < nl
            v = gelu_exact.(v)
        end
    end
    return pw(p.q2, gelu_exact.(pw(p.q1, v)))
end
"""residual: 출력 = a + net(a) (한 걸음 사상).  conserve: 증분의 평균(k = 0 모드)을 지워 평균을 구조적으로 보존한다."""
function fno1d(p, a; grid = false, residual = false, conserve = false)
    out = fno1d_net(p, a; grid)
    if conserve
        out = out .- mean(out; dims = 2)
    end
    return residual ? a[1:size(out, 1), :, :] .+ out : out
end

"""(net(a), 방향 da 위의 도함수): 전방 모드를 층마다 손으로 전파한다. 선형층과 Fourier 층은 접선에 그대로 작용하고
(편향 없이), GELU에서는 dgelu(z)·dz가 된다. 시간 조건부 모형의 ∂/∂t(PINO의 잔차)와 학습된 한 걸음 사상의 야코비안에 쓴다."""
function fno1d_jvp(p, a, da)
    v, dv = pw(p.lift, a), pw0(p.lift, da)
    nl = length(p.layers)
    for i in 1:nl
        l = p.layers[i]
        z = spectral1d(l.K, v) .+ pw(l.W, v); dz = spectral1d(l.K, dv) .+ pw0(l.W, dv)
        if i < nl
            v, dv = gelu_exact.(z), dgelu.(z) .* dz
        else
            v, dv = z, dz
        end
    end
    h = pw(p.q1, v)
    return pw(p.q2, gelu_exact.(h)), pw0(p.q2, dgelu.(h) .* pw0(p.q1, dv))
end

# ---------------------------------------------------------------- FNO-2D
"가중치 블록 (C′, C, m, m)의 축은 (출력, 입력, k2, k1): PyTorch의 (k1, k2, 입력, 출력)을 뒤집은 것. 초기화는 s · U(0, 1)"
function init_fno2d(rng; modes = 12, width = 20, depth = 4, in_ch = 1, proj = 64)
    s = 1.0f0 / (width * width)
    blk() = s .* rand(rng, Float32, width, width, modes, modes)
    layer() = (K = (Pr = blk(), Pi = blk(), Nr = blk(), Ni = blk()), W = linear(rng, width, width))
    return (lift = linear(rng, in_ch + 2, width), layers = Tuple(layer() for _ in 1:depth),
            q1 = linear(rng, width, proj), q2 = linear(rng, proj, 1))
end

"한 블록의 모드별 행렬곱: vh (C, m, m, B) 복소, 가중치 (C′, C, m, m)"
function modemul(Rr, Ri, vh)
    C, m2, m1, B = size(vh)
    x = reshape(permutedims(vh, (1, 4, 2, 3)), C, B, m2 * m1)           # (C, B, 모드)
    xr, xi = real(x), imag(x)
    Wr, Wi = reshape(Rr, size(Rr, 1), C, :), reshape(Ri, size(Ri, 1), C, :)
    yr = batched_mul(Wr, xr) .- batched_mul(Wi, xi)
    yi = batched_mul(Wr, xi) .+ batched_mul(Wi, xr)
    return permutedims(reshape(complex.(yr, yi), :, B, m2, m1), (1, 3, 4, 2))   # (C′, m, m, B)
end
"""v (C, N2, N1, B). rfft(v, (2, 3))은 첫 변환 축(N2)을 반으로 줄인다: (C, N2÷2+1, N1, B).
남기는 모드: k2 ∈ [0, m), k1 ∈ [0, m) (가중치 P)와 k1 ∈ [-m, 0) (가중치 N)."""
function spectral2d(l, v)
    C, N2, N1, B = size(v); m = size(l.Pr, 3); nf = N2 ÷ 2 + 1
    vh = rfft(v, (2, 3))
    yp = modemul(l.Pr, l.Pi, vh[:, 1:m, 1:m, :])
    yn = modemul(l.Nr, l.Ni, vh[:, 1:m, N1-m+1:N1, :])
    Co = size(yp, 1)
    top = cat(yp, zeros(eltype(yp), Co, m, N1 - 2m, B), yn; dims = 3)   # (C′, m, N1, B)
    return irfft(cat(top, zeros(eltype(yp), Co, nf - m, N1, B); dims = 2), N2, (2, 3))
end

"좌표 채널 둘 (2, N2, N1, B): 첫째는 N1 축의 좌표, 둘째는 N2 축의 좌표 (PyTorch의 g1, g2)"
function gridchannels(T, N2, N1, B)
    g1 = repeat(reshape(T.((0:N1-1) ./ N1), 1, 1, N1, 1), 1, N2, 1, B)
    g2 = repeat(reshape(T.((0:N2-1) ./ N2), 1, N2, 1, 1), 1, 1, N1, B)
    return cat(g1, g2; dims = 1)
end
ChainRulesCore.@non_differentiable gridchannels(T::Any, N2::Any, N1::Any, B::Any)

"u (N2, N1, B) → 다음 상태 u + Q(…). conserve면 증분의 평균을 지운다(평균이 구조적으로 보존된다)"
function fno2d(p, u; conserve = false)
    N2, N1, B = size(u)
    v = pw(p.lift, cat(reshape(u, 1, N2, N1, B), gridchannels(eltype(u), N2, N1, B); dims = 1))
    nl = length(p.layers)
    for i in 1:nl
        l = p.layers[i]
        v = spectral2d(l.K, v) .+ pw(l.W, v)
        if i < nl
            v = gelu_exact.(v)
        end
    end
    d = dropdims(pw(p.q2, gelu_exact.(pw(p.q1, v))); dims = 1)
    if conserve
        d = d .- mean(d; dims = (1, 2))
    end
    return u .+ d
end

"Float32로 학습한 매개변수를 Float64로 (구조 점검은 배정밀도 추론으로 한다). destructure의 re는 원래 정밀도로 되돌리므로 fmap을 쓴다"
to64(p) = Functors.fmap(x -> x isa AbstractArray{<:AbstractFloat} ? Float64.(x) : x, p)

# ---------------------------------------------------------------- 학습 루프
"""에폭 단위 학습: Adam + one-cycle(코사인, 처음 pct가 워밍업). lossfn(p, idx)는 표본 지표 idx의 미니배치 손실.
wd는 PyTorch Adam(weight_decay)의 결합 L2, clip은 전역 기울기 노름 자르기. (매개변수, 에폭별 평균 손실)을 돌려준다."""
function fit_epochs(p0, lossfn, ntrain; epochs, batch, lr, wd = 0.0, clip = Inf, pct = 0.1, seed = 0, drop_last = false)
    rng = Xoshiro(seed); θ, re = Optimisers.destructure(p0)
    opt = Optimisers.setup(Optimisers.Adam(lr), θ)
    nb = drop_last ? ntrain ÷ batch : cld(ntrain, batch)
    steps = epochs * nb; s = 0; hist = Float64[]
    for ep in 1:epochs
        perm = randperm(rng, ntrain); tot = 0.0
        for b in 1:nb
            idx = perm[(b-1)*batch+1:min(b * batch, ntrain)]
            L, gs = Zygote.withgradient(t -> lossfn(re(t), idx), θ)
            opt, θ = adam_step!(opt, θ, gs[1]; lr = onecycle_lr(lr, s, steps; pct), wd, clip); s += 1
            tot += L * length(idx)
        end
        push!(hist, tot / (drop_last ? nb * batch : ntrain))
    end
    return re(θ), hist
end

"""갱신 횟수 단위 학습: 매 걸음 batchfn(rng)로 미니배치를 뽑아 lossfn(p, batch)를 줄인다. sched(s)는 학습률 일정
(기본은 one-cycle). 미니배치를 뽑는 난수는 미분 밖에 둔다."""
function fit_steps(p0, lossfn, batchfn; steps, lr, sched = nothing, pct = 0.1, wd = 0.0, clip = Inf, seed = 0, log = 0)
    rng = Xoshiro(seed); θ, re = Optimisers.destructure(p0)
    opt = Optimisers.setup(Optimisers.Adam(lr), θ); t0 = time()
    for s in 0:steps-1
        batch = batchfn(rng)
        L, gs = Zygote.withgradient(t -> lossfn(re(t), batch), θ)
        opt, θ = adam_step!(opt, θ, gs[1]; lr = sched === nothing ? onecycle_lr(lr, s, steps; pct) : sched(s), wd, clip)
        log > 0 && (s + 1) % log == 0 && @printf("    step %5d  loss %.4f  (%.0f s)\n", s + 1, L, time() - t0)
    end
    return re(θ)
end
