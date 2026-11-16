# family.jl — §9.4 Lab(9.13, 9.14)의 공용 부분 (labs/94_common.py)
# 주기 영역 x ∈ [0, 2π) 위의 방정식 족  u_t + c u_x + β (u²/2)_x = ν u_xx + ρ (u - u³)
#   · 솔버: 의사 스펙트럴 ETDRK4 (Kassam–Trefethen의 경로 적분 계수), 2/3 규칙
#   · 모형: 시간 조건부 FNO-1D, 강제 초기조건 u = u0 + t · NN(u0, 계수, t)
#   · 잔차: 예측의 라벨 없는 PDE 잔차 (x는 스펙트럴 미분, t는 손으로 전파한 전방 모드 접선)
# 배치 약속: 장 (N, B), 계수 (4, B) = (ν, c, β, ρ), 시각 (B). 궤적 자료는 (N, 41, B).
include("ch9_common.jl")
include("fno.jl")

# ---------------------------------------------------------------- 1. 자료
"무작위 주기 초기조건: 평균 + Σ_{k ≤ kmax} k^-decay (a_k cos kx + b_k sin kx), max|u0 - 평균| = amp · U(0.5, 1), 평균 ~ U(-0.3, 0.3)"
function sample_ic(rng, n, N; kmax = 5, amp = 1.0, decay = 1.0)
    x = (0:N-1) .* (2pi / N); k = 1:kmax
    a = randn(rng, kmax, n) ./ k .^ decay; b = randn(rng, kmax, n) ./ k .^ decay
    u = cos.(x .* k') * a .+ sin.(x .* k') * b                           # (N, n)
    u .*= (amp .* (0.5 .+ 0.5 .* rand(rng, 1, n))) ./ maximum(abs, u; dims = 1)
    return u .+ (0.6 .* rand(rng, 1, n) .- 0.3)
end

"과제의 사전분포. :pre는 사전학습 족(ρ = 0), :lownu는 범위 아래의 ν, :react는 본 적 없는 반응항 ρ ~ U[1, 3]. (4, n)"
function sample_coeffs(rng, n, kind = :pre)
    logu(lo, hi) = exp.(log(lo) .+ (log(hi) - log(lo)) .* rand(rng, n))
    nu = logu(0.05, 0.2); c = 2 .* rand(rng, n) .- 1; beta = rand(rng, n); rho = zeros(n)
    kind == :lownu && (nu = logu(0.02, 0.04))
    kind == :react && (rho = 1 .+ 2 .* rand(rng, n))
    return permutedims(hcat(nu, c, beta, rho))
end

"""ETDRK4: u_t = Lin u + N(u), Lin = -ν k² - i c k, N = -β (u²/2)_x + ρ (u - u³). u0 (N, B), coef (4, B).
(N, n_save + 1, B)의 스냅숏(t = 0 포함)을 돌려준다. φ 함수는 단위원 위 64점의 경로 적분으로 구한다(Lin이 복소수라 온 원)."""
function etdrk4(u0, coef; T = 1.0, dt = 5e-3, n_save = 40, dealias = true)
    N, B = size(u0); k = collect(0.0:N÷2)
    nu, c, beta, rho = (reshape(Float64.(coef[i, :]), 1, B) for i in 1:4)
    Lin = -nu .* k .^ 2 .- im .* c .* k                                  # (nk, B)
    E, E2 = exp.(dt .* Lin), exp.(dt .* Lin ./ 2)
    M = 64; r = reshape(exp.(2im * pi .* ((1:M) .- 0.5) ./ M), 1, 1, M)
    LR = dt .* Lin .+ r                                                  # (nk, B, M)
    avg(A) = dropdims(mean(A; dims = 3); dims = 3)
    Q = dt .* avg((exp.(LR ./ 2) .- 1) ./ LR)
    f1 = dt .* avg((-4 .- LR .+ exp.(LR) .* (4 .- 3 .* LR .+ LR .^ 2)) ./ LR .^ 3)
    f2 = dt .* avg((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR .^ 3)
    f3 = dt .* avg((-4 .- 3 .* LR .- LR .^ 2 .+ exp.(LR) .* (4 .- LR)) ./ LR .^ 3)
    mask = dealias ? Float64.(k .<= N ÷ 3) : ones(length(k))
    function Nl(v)
        u = irfft(v, N, 1)
        return (-0.5im .* k .* beta .* rfft(u .* u, 1) .+ rho .* rfft(u .- u .^ 3, 1)) .* mask
    end
    v = rfft(u0, 1); steps = round(Int, T / dt); every = steps ÷ n_save
    @assert every * n_save == steps
    out = zeros(N, n_save + 1, B); out[:, 1, :] = u0
    for s in 1:steps
        Nv = Nl(v); a = E2 .* v .+ Q .* Nv
        Na = Nl(a); b = E2 .* v .+ Q .* Na
        Nb = Nl(b); cc = E2 .* a .+ Q .* (2 .* Nb .- Nv)
        Nc = Nl(cc)
        v = E .* v .+ Nv .* f1 .+ 2 .* (Na .+ Nb) .* f2 .+ Nc .* f3
        s % every == 0 && (out[:, s÷every+1, :] = irfft(v, N, 1))
    end
    return out
end

"256점에서 풀고 128점으로 저장한다. (u = (128, 41, n) Float32, coef = (4, n) Float32), 솔버 시간(초)도 함께"
function make_dataset(n, seed; kind = :pre, n_solver = 256, n_grid = 128, amp = 1.0, kmax = 5, chunk = 500)
    rng = Xoshiro(seed)
    u0 = sample_ic(rng, n, n_solver; kmax, amp); coef = sample_coeffs(rng, n, kind)
    t0 = time()
    u = cat([etdrk4(u0[:, i:min(i + chunk - 1, n)], coef[:, i:min(i + chunk - 1, n)]) for i in 1:chunk:n]...; dims = 3)
    return (u = Float32.(u[1:n_solver÷n_grid:end, :, :]), coef = Float32.(coef)), time() - t0
end

# ---------------------------------------------------------------- 2. 모형: 시간 조건부 FNO-1D (TFNO)
"lift 6 → width (채널: u0, log(ν/0.1), c, β, ρ, t), 잔차 연결이 있는 Fourier 블록, 투영 width → 64 → 1"
function init_tfno(rng; width = 32, modes = 16, depth = 4)
    s = 1.0f0 / width / sqrt(2.0f0)                                      # R = 복소 정규 / width
    blk() = (K = (Rr = s .* randn(rng, Float32, width, width, modes), Ri = s .* randn(rng, Float32, width, width, modes)),
             W = linear(rng, width, width))
    lift = linear(rng, 6, width)
    lift.W[:, 5] .= 0                                                    # 사전학습에 없는 계수(ρ)는 0인 열로 들어온다
    return (lift = lift, blocks = Tuple(blk() for _ in 1:depth), q1 = linear(rng, width, 64), q2 = linear(rng, 64, 1))
end

"입력 채널 (6, N, B)"
function tfno_input(u0, coef, t)
    N, B = size(u0)
    cf = vcat(log.(coef[1:1, :] ./ 0.1f0), coef[2:4, :])
    return vcat(reshape(u0, 1, N, B), repeat(reshape(cf, 4, 1, B), 1, N, 1), repeat(reshape(t, 1, 1, B), 1, N, 1))
end
"u_θ(x, t; u0, 계수) = u0 + t · NN: 초기조건이 정확히 성립한다. u0 (N, B) → (N, B)"
function tfno(p, u0, coef, t)
    v = pw(p.lift, tfno_input(u0, coef, t))
    for i in 1:length(p.blocks)
        l = p.blocks[i]
        v = v .+ gelu_exact.(spectral1d(l.K, v) .+ pw(l.W, v))            # σ(K v + W v), 잔차 연결
    end
    return u0 .+ reshape(t, 1, :) .* dropdims(pw(p.q2, gelu_exact.(pw(p.q1, v))); dims = 1)
end

"(u, ∂u/∂t): 시간 방향 접선을 층마다 전파한다. 입력의 t 채널의 접선은 1, 나머지는 0이다"
function tfno_jvp(p, u0, coef, t)
    N, B = size(u0); T = eltype(u0)
    v = pw(p.lift, tfno_input(u0, coef, t))
    dv = pw0(p.lift, vcat(zeros(T, 5, N, B), ones(T, 1, N, B)))
    for i in 1:length(p.blocks)
        l = p.blocks[i]
        z = spectral1d(l.K, v) .+ pw(l.W, v); dz = spectral1d(l.K, dv) .+ pw0(l.W, dv)
        v, dv = v .+ gelu_exact.(z), dv .+ dgelu.(z) .* dz
    end
    h = pw(p.q1, v)
    nn = dropdims(pw(p.q2, gelu_exact.(h)); dims = 1); dnn = dropdims(pw0(p.q2, dgelu.(h) .* pw0(p.q1, dv)); dims = 1)
    tt = reshape(t, 1, :)
    return u0 .+ tt .* nn, nn .+ tt .* dnn
end

# ---------------------------------------------------------------- 3. 학습, 예측, 오차
"무작위 (궤적, 시각) 쌍의 미니배치: (u0, 계수, t_j = j/40, 목표)"
function sample_batch(rng, data, bs)
    n = size(data.u, 3); nt = size(data.u, 2)
    i = rand(rng, 1:n, bs); j = rand(rng, 2:nt, bs)                      # j ≥ 2: t = 0은 빼고
    target = reduce(hcat, [data.u[:, j[b], i[b]] for b in 1:bs])
    return data.u[:, 1, i], data.coef[:, i], Float32.((j .- 1) ./ (nt - 1)), target
end
data_loss(p, (u0, coef, t, y)) = rel_loss(tfno(p, u0, coef, t), y)
train_tfno(p, data; steps, bs = 64, lr = 3e-3, seed = 0, log = 0) =
    fit_steps(p, data_loss, rng -> sample_batch(rng, data, bs); steps, lr, pct = 0.05, seed, log)

"스냅숏 지표 tidx(1부터; t = (j - 1)/40)에서의 예측 (N, length(tidx), n)"
function predict(p, data, tidx; bs = 256)
    n = size(data.u, 3); nt = size(data.u, 2)
    out = [reduce(hcat, [tfno(p, data.u[:, 1, i:min(i + bs - 1, n)], data.coef[:, i:min(i + bs - 1, n)],
                              fill(Float32((j - 1) / (nt - 1)), length(i:min(i + bs - 1, n)))) for i in 1:bs:n]) for j in tidx]
    return stack(out; dims = 2)
end
const T_IDX = collect(5:4:41)                                            # t = 0.1, 0.2, …, 1.0
"궤적마다의 시공간 상대 L2 오차 (t = 0.1..1.0 블록)"
traj_error(p, data; tidx = T_IDX) = rel_each(predict(p, data, tidx), data.u[:, tidx, :])

# ---------------------------------------------------------------- 4. 라벨 없는 PDE 잔차
"선형항의 기호 -ν k² - i c k (nk, B)와 보존형 이류항의 기호 (i k/2)·마스크 (nk). 자료에서만 정해지는 상수다"
function symbols(coef, N, T; dealias = true)
    k = T.(0:N÷2)
    return (-coef[1:1, :] .* k .^ 2 .- im .* coef[2:2, :] .* k), (T(0.5) * im) .* k .* (dealias ? T.(k .<= N ÷ 3) : one.(k))
end
ChainRulesCore.@non_differentiable symbols(coef::Any, N::Any, T::Any)
"ν u_xx - c u_x - β (u²/2)_x + ρ (u - u³), 스펙트럴 미분. u (N, B), coef (4, B)"
function rhs(u, coef)
    N = size(u, 1)
    lin, adv = symbols(coef, N, eltype(u))
    return irfft(lin .* rfft(u, 1), N, 1) .- coef[3:3, :] .* irfft(adv .* rfft(u .* u, 1), N, 1) .+ coef[4:4, :] .* (u .- u .^ 3)
end
"(u, r = ∂u/∂t - RHS(u)). 시간 도함수는 정확하다(접선 전파)"
function pde_residual(p, u0, coef, t)
    u, ut = tfno_jvp(p, u0, coef, t)
    return u, ut .- rhs(u, coef)
end
"궤적마다의 RMS 잔차 (t = 0.1..1.0 블록; 라벨을 쓰지 않는다). predict는 여러 모형의 평균이어도 된다: jvp(u0, coef, t) → (u, u_t)"
function residual_norms(jvp, data; tidx = T_IDX, bs = 256)
    n = size(data.u, 3); nt = size(data.u, 2); r2 = zeros(n)
    for j in tidx, i in 1:bs:n
        ii = i:min(i + bs - 1, n)
        u, ut = jvp(data.u[:, 1, ii], data.coef[:, ii], fill(Float32((j - 1) / (nt - 1)), length(ii)))
        r2[ii] .+= vec(mean(abs2, ut .- rhs(u, data.coef[:, ii]); dims = 1))
    end
    return sqrt.(r2 ./ length(tidx))
end

# ---------------------------------------------------------------- 5. LoRA: 모든 점별 선형층에 저랭크 보정 ΔW = B A
"A ~ N(0, 1/fan_in), B = 0이므로 시작할 때 ΔW = 0 (Lab 7.17과 같은 구성). Fourier 가중치 R은 얼려 둔다"
lora_pair(rng, l, r) = (A = randn(rng, Float32, r, size(l.W, 2)) ./ sqrt(Float32(size(l.W, 2))), B = zeros(Float32, size(l.W, 1), r))
init_lora(rng, p, r) = (lift = lora_pair(rng, p.lift, r), blocks = map(b -> lora_pair(rng, b.W, r), p.blocks),
                        q1 = lora_pair(rng, p.q1, r), q2 = lora_pair(rng, p.q2, r))
addlora(l, a) = (W = l.W .+ a.B * a.A, b = l.b)
"얼린 매개변수 p에 어댑터 a를 더한 모형"
with_lora(p, a) = (lift = addlora(p.lift, a.lift), blocks = map((b, ab) -> (K = b.K, W = addlora(b.W, ab)), p.blocks, a.blocks),
                   q1 = addlora(p.q1, a.q1), q2 = addlora(p.q2, a.q2))
