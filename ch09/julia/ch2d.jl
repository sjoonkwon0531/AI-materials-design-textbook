# ch2d.jl — Ch.2 Lab 2.14 (Derivation Box 2.22)의 Cahn–Hilliard 솔버를 배치 형태로 다시 쓴 것 (labs/96_ch_solver.py)
#   ∂c/∂t = ∇²(c³ - c) - ε² ∇⁴c,   주기 정사각형 (0, L)²
#   ĉⁿ⁺¹ = [(1 + dt S k²) ĉⁿ - dt k² FFT(c³ - c)ⁿ] / (1 + dt S k² + dt ε² k⁴)
# k = 0에서 분자와 분모가 모두 1이므로 평균(질량)은 스킴의 구조로 보존된다.
# 배치 약속은 ns2d.jl과 같다: 장 (N, N, B), 스펙트럼 (N÷2+1, N, B) (첫 축이 반쪽).
using FFTW, Random, Statistics

struct CH2D
    N::Int
    L::Float64
    eps::Float64
    dt::Float64
    S::Float64
    dx::Float64
    KX::Matrix{Float64}
    KY::Matrix{Float64}
    K2::Matrix{Float64}
    gain::Matrix{Float64}
    denom::Matrix{Float64}
    wt::Matrix{Float64}                                               # 반쪽 스펙트럼의 합을 온 스펙트럼의 합으로 바꾸는 가중치
end
function CH2D(; N = 64, L = 2pi, eps = 0.10, dt = 0.05, S = 2.0)
    KY = repeat(reshape(collect(0.0:N÷2) .* (2pi / L), :, 1), 1, N)
    KX = repeat(reshape(collect(fftfreq(N, N)) .* (2pi / L), 1, :), N ÷ 2 + 1, 1)
    K2 = KX .^ 2 .+ KY .^ 2
    wt = fill(2.0, size(K2)); wt[1, :] .= 1.0
    iseven(N) && (wt[end, :] .= 1.0)
    return CH2D(N, L, eps, dt, S, L / N, KX, KY, K2, 1 .+ dt * S .* K2, 1 .+ dt * S .* K2 .+ dt * eps^2 .* K2 .^ 2, wt)
end

"c (N, N[, B])를 nsteps걸음 전진 (Box 2.22)"
function advance(s::CH2D, c, nsteps)
    ch = rfft(c, (1, 2))
    for _ in 1:nsteps
        ch = (s.gain .* ch .- s.dt .* s.K2 .* rfft(c .^ 3 .- c, (1, 2))) ./ s.denom
        c = irfft(ch, s.N, (1, 2))
    end
    return c
end
"(N, N, n_rec + 1, B)"
function trajectory(s::CH2D, c0, n_rec, steps_per_rec)
    out = zeros(s.N, s.N, n_rec + 1, size(c0, 3)); out[:, :, 1, :] = c0; c = c0
    for r in 1:n_rec
        c = advance(s, c, steps_per_rec); out[:, :, r+1, :] = c
    end
    return out
end

"자유에너지 F = Σ [(c² - 1)²/4 + ε² |∇c|²/2] dx² (스펙트럴 기울기). 표본마다 하나"
function free_energy(s::CH2D, c)
    ch = rfft(c, (1, 2))
    cx, cy = irfft(im .* s.KX .* ch, s.N, (1, 2)), irfft(im .* s.KY .* ch, s.N, (1, 2))
    return vec(sum(0.25 .* (c .^ 2 .- 1) .^ 2 .+ 0.5 * s.eps^2 .* (cx .^ 2 .+ cy .^ 2); dims = (1, 2))) .* s.dx^2
end
"특성 길이 L(t) = 2π/⟨k⟩, ⟨k⟩ = Σ|k| S(k) / Σ S(k), S(k) = |FFT(c - 평균)|²"
function char_length(s::CH2D, c)
    Sk = abs2.(rfft(c .- mean(c; dims = (1, 2)), (1, 2))) .* s.wt
    return vec(2pi .* sum(Sk; dims = (1, 2)) ./ sum(sqrt.(s.K2) .* Sk; dims = (1, 2)))
end
"급랭: 평균 조성 m 둘레의 균등 잡음 (진폭 amp). m은 스칼라 또는 길이 B의 벡터"
random_ic(rng, B, N, m; amp = 0.1) = reshape(m .* ones(B), 1, 1, B) .+ amp .* (2 .* rand(rng, N, N, B) .- 1)
"질량 표류 max_t |⟨c⟩(t) - ⟨c⟩(0)|: 궤적 (N, N, T, B)의 표본마다"
mass_drift(traj) = (mu = dropdims(mean(traj; dims = (1, 2)); dims = (1, 2)); vec(maximum(abs.(mu .- mu[1:1, :]); dims = 1)))
"log t – log L의 최소제곱 기울기 (lo ≤ t ≤ hi)"
function growth_slope(t, L; lo = 5.0, hi = 50.0)
    f = (t .>= lo) .& (t .<= hi)
    return ([ones(count(f)) log.(t[f])] \ log.(L[f]))[2]
end
