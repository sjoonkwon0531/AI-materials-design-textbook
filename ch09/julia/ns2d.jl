# ns2d.jl — 2차원 Navier–Stokes(와도 꼴)의 의사 스펙트럴 솔버 (labs/95_ns_solver.py)
#   ∂w/∂t + u·∇w = ν ∇²w + f,  u = (ψ_y, -ψ_x),  ∇²ψ = -w,  단위 토러스 (0, 1)²
# 외력과 초기조건은 Li 등(2021)과 같다: f = 0.1 (sin 2π(x + y) + cos 2π(x + y)), w0 ~ N(0, 7^{3/2} (-Δ + 49 I)^{-2.5}).
# 시간: 확산항은 Crank–Nicolson, 이류항은 2차 Adams–Bashforth(첫 걸음은 Euler), 이차항에 2/3 규칙.
# 배치 약속: 장은 (N, N, B) = [y 지표, x 지표, 표본] (numpy (B, x, y)의 축을 뒤집은 것).
# rfft(w, (1, 2))는 첫 축(y)을 반으로 줄이므로 스펙트럼은 (N÷2+1, N, B)이고, 반쪽 축이 ky, 온 축이 kx다.
using FFTW, Random, Statistics

struct NS2D
    N::Int
    nu::Float64
    dt::Float64
    ikx::Matrix{ComplexF64}
    iky::Matrix{ComplexF64}
    lap::Matrix{Float64}
    inv_lap::Matrix{Float64}
    mask::Matrix{Float64}
    fh::Matrix{ComplexF64}
    cn_num::Matrix{Float64}
    cn_den::Matrix{Float64}
end

forcing(N) = [0.1 * (sin(2pi * (x + y)) + cos(2pi * (x + y))) for y in (0:N-1) ./ N, x in (0:N-1) ./ N]

function NS2D(N; nu = 1e-3, dt = 1e-3, force = true, dealias = true)
    KY = repeat(reshape(collect(0.0:N÷2), :, 1), 1, N)                 # 반쪽 축 (dim 1)
    KX = repeat(reshape(collect(fftfreq(N, N)), 1, :), N ÷ 2 + 1, 1)   # 온 축 (dim 2): 0..N/2-1, -N/2..-1
    lap = 4pi^2 .* (KX .^ 2 .+ KY .^ 2)
    inv_lap = [l > 0 ? 1 / l : 0.0 for l in lap]
    kmax = N ÷ 2
    mask = dealias ? Float64.((abs.(KX) .<= (2 / 3) * kmax) .& (abs.(KY) .<= (2 / 3) * kmax)) : ones(size(lap))
    fh = force ? rfft(forcing(N), (1, 2)) : zeros(ComplexF64, size(lap))
    return NS2D(N, nu, dt, 2im * pi .* KX, 2im * pi .* KY, lap, inv_lap, mask, fh, 1 .- 0.5 * dt * nu .* lap, 1 .+ 0.5 * dt * nu .* lap)
end

"-(u·∇w)의 Fourier 변환, 2/3 규칙으로 dealias"
function nonlinear(s::NS2D, wh)
    psih = wh .* s.inv_lap
    ir(a) = irfft(a, s.N, (1, 2))
    u, v = ir(s.iky .* psih), ir(-s.ikx .* psih)
    wx, wy = ir(s.ikx .* wh), ir(s.iky .* wh)
    return -s.mask .* rfft(u .* wx .+ v .* wy, (1, 2))
end

"스펙트럼 wh를 nsteps걸음 전진. n_old는 AB2의 이력(없으면 첫 걸음이 Euler). (wh, n_old)를 돌려준다"
function steps!(s::NS2D, wh, n_old, nsteps)
    for _ in 1:nsteps
        n_now = nonlinear(s, wh)
        adv = n_old === nothing ? n_now : 1.5 .* n_now .- 0.5 .* n_old
        wh = (s.cn_num .* wh .+ s.dt .* (adv .+ s.fh)) ./ s.cn_den
        n_old = n_now
    end
    return wh, n_old
end
"w (N, N, B)를 nsteps·dt만큼 전진 (AB2를 Euler 한 걸음으로 다시 시작)"
advance(s::NS2D, w, nsteps) = irfft(steps!(s, rfft(w, (1, 2)), nothing, nsteps)[1], s.N, (1, 2))
"(N, N, n_rec + 1, B): t = 0, steps_per_rec·dt, … 의 w. AB2 이력은 기록 사이에도 이어진다"
function trajectory(s::NS2D, w0, n_rec, steps_per_rec)
    out = zeros(s.N, s.N, n_rec + 1, size(w0, 3)); out[:, :, 1, :] = w0
    wh = rfft(w0, (1, 2)); n_old = nothing
    for r in 1:n_rec
        wh, n_old = steps!(s, wh, n_old, steps_per_rec)
        out[:, :, r+1, :] = irfft(wh, s.N, (1, 2))
    end
    return out
end

"Gauss 무작위장: 공분산 τ^{2α-2} (-Δ + τ²)^{-α}, 평균 0 (FNO 자료 생성 코드의 GaussianRF와 같은 구성). (N, N, B)"
function grf2d(rng, B, N; alpha = 2.5, tau = 7.0)
    k = collect(fftfreq(N, N))
    sq = N^2 * sqrt(2.0) * tau^(0.5 * (2alpha - 2)) .* (4pi^2 .* (k .^ 2 .+ (k') .^ 2) .+ tau^2) .^ (-alpha / 2)
    sq[1, 1] = 0.0
    return real.(ifft(sq .* (randn(rng, N, N, B) .+ im .* randn(rng, N, N, B)), (1, 2)))
end

"해상도 N → M: Fourier 계수를 자르거나 0으로 채운다 (작은 격자의 Nyquist 모드는 버려 결과가 실수가 되게)"
function spectral_resample(w, M)
    N = size(w, 1); M == N && return copy(w)
    wh = fft(w, (1, 2)); m = min(N, M) ÷ 2
    src = vcat(1:m, N-m+2:N); dst = vcat(1:m, M-m+2:M); rest = ntuple(_ -> Colon(), ndims(w) - 2)
    out = zeros(ComplexF64, M, M, size(w)[3:end]...)
    out[dst, dst, rest...] = wh[src, src, rest...]
    return real.(ifft(out, (1, 2))) .* (M / N)^2
end

"표본마다의 상대 L2 오차 (앞의 두 축 위에서)"
rel2d(a, b) = dropdims(sqrt.(sum(abs2, a .- b; dims = (1, 2)) ./ sum(abs2, b; dims = (1, 2))); dims = (1, 2))
