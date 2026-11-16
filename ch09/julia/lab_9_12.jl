# lab_9_12.jl — Lab 9.12: Prior에 맞춘 문맥 내 연산자 대 정확한 Bayes 사후 예측 (labs/94a_incontext.py)
# 과제의 사전분포: u_t + c u_x = ν u_xx on [0, 2π), T = 1, ν ~ logU[0.02, 0.5], c ~ U[-1.5, 1.5].
# Fourier 공간에서 해 연산자는 대각이다: û_T(k) = m_k û_0(k), m_k = exp(-ν k² - i c k), k = 1..8.
# 문맥: K개의 쌍 (u0_i, y_i), y_i = u_T,i + 복소 Gauss 잡음(모드마다 σ = 0.2). 질의: 새 u0에서 u_T를 예측. (ν, c)는 주지 않는다.
# 비교: 사전 평균(K = 0) | 모드별 최소제곱 S_k/P_k | 정확한 Bayes(격자 사후분포) | 신경망. 솔버가 없는 Lab이라 전부 옮겼다.
# 배치 약속: 과제가 열이다. 문맥 x, y는 (M, K, n), 질의와 곱수는 (M, n). 신경망은 Lux.jl MLP(24-256-256-256-16, GELU).
include("ch9_common.jl")
using Lux

const M, SIG, KTR = 8, 0.2, 8                                           # 모드 수, 잡음, 학습 때의 최대 문맥 크기
const KK = collect(1:M); const SK = 1.0 ./ KK                           # |û_0(k)|의 사전 표준편차
const NU_LO, NU_HI, C_MAX = 0.02, 0.5, 1.5
cn(rng, dims...) = (randn(rng, dims...) .+ im .* randn(rng, dims...)) ./ sqrt(2)

"문맥 입력 x (M, K, n), 잡음 섞인 출력 y, 질의 xq (M, n), 참 곱수 m (M, n)"
function sample_tasks(rng, n, K; nu = nothing, c = nothing)
    nu === nothing && (nu = exp.(log(NU_LO) .+ (log(NU_HI) - log(NU_LO)) .* rand(rng, n)))
    c === nothing && (c = C_MAX .* (2 .* rand(rng, n) .- 1))
    m = exp.(-KK .^ 2 .* nu' .- im .* KK .* c')                         # (M, n)
    x = cn(rng, M, K, n) .* SK; xq = cn(rng, M, n) .* SK
    y = reshape(m, M, 1, n) .* x .+ SIG .* cn(rng, M, K, n)
    return x, y, xq, m
end
"충분통계량 S_k = Σ_i y_ik conj(x_ik), P_k = Σ_i |x_ik|² (가능도에 필요한 전부). (M, n)"
suff_stats(x, y) = (dropdims(sum(y .* conj.(x); dims = 2); dims = 2), dropdims(sum(abs2, x; dims = 2); dims = 2))
"신경망 입력 (3M, n): 능형 추정 S/(P + σ²)의 실수부·허수부와 로그 정밀도 log(1 + P/σ²)"
features(S, P) = (mh = S ./ (P .+ SIG^2); Float32.(vcat(real.(mh), imag.(mh), log1p.(P ./ SIG^2))))

# ---------------------------------------------------------------- (log ν, c) 격자 위의 정확한 Bayes
const G_NU = exp.(range(log(NU_LO), log(NU_HI); length = 160)); const G_C = collect(range(-C_MAX, C_MAX; length = 1501))
const GNU = vec([nu for c in G_C, nu in G_NU]); const GC = vec([c for c in G_C, nu in G_NU])        # 격자점 240,160개
const M_GRID = exp.(-KK .^ 2 .* GNU' .- im .* KK .* GC')                                             # (M, G)
const A_GRID = vcat(2 .* real.(M_GRID), 2 .* imag.(M_GRID), -abs2.(M_GRID)) ./ SIG^2                 # (3M, G)
"문맥이 주어졌을 때 m_k의 사후 평균: Σ_g w_g m_k(g), w_g ∝ 사전분포 x 가능도"
function bayes_multiplier(S, P; chunk = 25)
    out = zeros(ComplexF64, size(S))
    for i in 1:chunk:size(S, 2)
        ii = i:min(i + chunk - 1, size(S, 2))
        ll = A_GRID' * vcat(real.(S[:, ii]), imag.(S[:, ii]), P[:, ii])   # 로그 가능도(상수 차이), (G, b)
        w = exp.(ll .- maximum(ll; dims = 1)); w ./= sum(w; dims = 1)
        out[:, ii] = M_GRID * w
    end
    return out
end

# ---------------------------------------------------------------- prior에 맞춘 신경망: 문맥 통계량 → Fourier 곱수 R(k)
tinit(nin) = (rng, dims...) -> (rand(rng, Float32, dims...) .* 2.0f0 .- 1.0f0) ./ sqrt(Float32(nin))
tdense(nin, nout, act = identity) = Dense(nin => nout, act; init_weight = tinit(nin), init_bias = tinit(nin))
const NET = Chain(tdense(3M, 256, gelu_exact), tdense(256, 256, gelu_exact), tdense(256, 256, gelu_exact), tdense(256, 2M))
net_multiplier(ps, st, S, P) = (o = Float64.(first(NET(features(S, P), ps, st))); complex.(o[1:M, :], o[M+1:end, :]))

function main(; steps = 30000, bs = 256)
    rng = Xoshiro(0); ps, st = Lux.setup(Xoshiro(0), NET)
    θ, re = Optimisers.destructure(ps); opt = Optimisers.setup(Optimisers.Adam(1e-3), θ); t0 = time()
    for s in 0:steps-1
        K = rand(rng, 0:KTR)                                            # 문맥 크기 0..8, 매 걸음 새 과제 묶음
        x, y, xq, m = sample_tasks(rng, bs, K); S, P = suff_stats(x, y)
        F = features(S, P); xr, xi = Float32.(real.(xq)), Float32.(imag.(xq)); yq = m .* xq
        yr, yi = Float32.(real.(yq)), Float32.(imag.(yq))
        L, gs = Zygote.withgradient(θ) do t                             # 질의 라벨에 대한 제곱 오차 |R xq - m xq|²
            o = first(NET(F, re(t), st)); Rr, Ri = o[1:M, :], o[M+1:end, :]
            mean(sum(abs2, Rr .* xr .- Ri .* xi .- yr; dims = 1) .+ sum(abs2, Rr .* xi .+ Ri .* xr .- yi; dims = 1))
        end
        opt, θ = adam_step!(opt, θ, gs[1]; lr = cosine_lr(1e-3, s, steps))
        (s + 1) % 10000 == 0 && @printf("step %d: loss %.4f\n", s + 1, L)
    end
    ps = re(θ)
    @printf("prior fitting: %d steps x %d tasks = %.2f M tasks, %.0f s, %d parameters\n", steps, bs, steps * bs / 1e6, time() - t0, length(θ))

    function evaluate(n, K, seed; nu = nothing, c = nothing)
        x, y, xq, m = sample_tasks(Xoshiro(seed), n, K; nu, c); S, P = suff_stats(x, y); tru = m .* xq
        rel(mult) = sqrt(sum(abs2, mult .* xq .- tru) / sum(abs2, tru))
        mb = bayes_multiplier(S, P); mn = net_multiplier(ps, st, S, P)
        ls = K > 0 ? S ./ max.(P, 1e-12) : zero(S)
        return rel(ls), rel(mb), rel(mn), sqrt(sum(abs2, (mn .- mb) .* xq) / sum(abs2, tru))
    end
    println("\nin-prior test, 500 tasks (relative L2 error of the predicted u_T for a new u0):")
    println("   K   per-mode LS   exact Bayes   network   |network - Bayes|")
    for K in (0, 1, 2, 4, 8, 16, 32)
        r = evaluate(500, K, 100 + K)
        @printf("  %2d     %6.3f       %6.3f      %6.3f       %6.3f%s\n", K, r[1], r[2], r[3], r[4], K <= KTR ? "" : "   (K beyond the training range)")
    end
    x, y, xq, m = sample_tasks(Xoshiro(5), 500, 8); S, P = suff_stats(x, y)
    tb = @elapsed bayes_multiplier(S, P); tn = best_time(() -> net_multiplier(ps, st, S, P); rep = 20)
    @printf("cost per task (K = 8): grid Bayes %.2f ms (%d grid points), network %.1f us  -> ratio %.0f\n", 1e3 * tb / 500, length(GNU), 1e6 * tn / 500, tb / tn)
    println("\nout-of-prior test (K = 8, 500 tasks):")
    rg = Xoshiro(7)
    for (name, nu, c) in (("nu in [0.005, 0.015] (below the prior range)", exp.(log(0.005) .+ (log(0.015) - log(0.005)) .* rand(rg, 500)), nothing),
                          ("c in [2, 3] (outside the prior range)", nothing, 2 .+ rand(rg, 500)), ("in prior (reference)", nothing, nothing))
        r = evaluate(500, 8, 300; nu, c)
        @printf("   %-46s LS %.3f | Bayes(grid prior) %.3f | network %.3f\n", name, r[1], r[2], r[3])
    end
end
main()
