# spectral1d.jl — §9.3의 자료를 만드는 1차원 의사 스펙트럴 솔버 (labs/93_data.py)
#   burgers_solve : u_t + (u²/2)_x = ν u_xx              (x ∈ [0, 1), 주기)
#   ks_solve      : u_t + (u²/2)_x + u_xx + u_xxxx = 0    (x ∈ [0, L), 주기)
# Fourier 콜로케이션, 2/3 규칙, 선형항은 적분 인자로 정확히, 비선형항은 고전 RK4 (IFRK4).
# 배치 약속: 장은 (N, B), 즉 열이 초기조건이다(파이썬 (B, N)의 전치). FFT는 첫 축 위에서 한다: rfft(u, 1).
using FFTW, Random, Statistics

"û′ = lin·û + N̂(û). u0는 (N, B), lin은 길이 N÷2+1의 실수 기호. save_every > 0이면 (N, 스냅숏, B)를 돌려준다(t = 0 포함)"
function ifrk4(u0, lin, nonlin, dt, nsteps; save_every = 0)
    N = size(u0, 1); uh = rfft(u0, 1)
    E, E2 = exp.(lin .* dt), exp.(lin .* (dt / 2))
    out = [copy(u0)]
    for n in 1:nsteps
        k1 = nonlin(uh)
        k2 = nonlin(E2 .* (uh .+ 0.5dt .* k1))
        k3 = nonlin(E2 .* uh .+ 0.5dt .* k2)
        k4 = nonlin(E .* uh .+ dt .* E2 .* k3)
        uh = E .* uh .+ (dt / 6) .* (E .* k1 .+ 2 .* E2 .* (k2 .+ k3) .+ k4)
        save_every > 0 && n % save_every == 0 && push!(out, irfft(uh, N, 1))
    end
    return save_every > 0 ? stack(out; dims = 2) : irfft(uh, N, 1)
end

"2/3 규칙의 마스크: 파수 지표 0..N÷3만 남긴다 (N = 64면 0..21)"
dealias_mask(N) = Float64.((0:N÷2) .<= N ÷ 3)
"N̂(û) = -(ik/2) FFT[u²], 곱하기 전과 뒤에 마스크"
function conv_flux(N, k)
    mask = dealias_mask(N)
    return function (uh)
        u = irfft(uh .* mask, N, 1)
        return -0.5im .* k .* rfft(u .* u, 1) .* mask
    end
end

function burgers_solve(u0, nu, T, dt; save_every = 0)
    N = size(u0, 1); k = 2pi .* (0:N÷2)                                # 영역 길이 1
    return ifrk4(u0, -nu .* k .^ 2, conv_flux(N, k), dt, round(Int, T / dt); save_every)
end
function ks_solve(u0, L, T, dt; save_every = 0)
    N = size(u0, 1); k = (2pi / L) .* (0:N÷2)
    return ifrk4(u0, k .^ 2 .- k .^ 4, conv_flux(N, k), dt, round(Int, T / dt); save_every)
end

"""[0, 1) 위의 Gauss 무작위장, 스펙트럼 λ_k = σ² / ((2πk)² + τ²)^α (Li 등 2021의 Burgers 설정).
k ≥ 1의 진폭은 정확한 측도의 √2배다(본문과 파이썬 판이 같은 약속을 쓴다). (N, n) 배열."""
function grf(rng, n, N; sigma = 25.0, tau = 5.0, alpha = 2.0, kmax = nothing)
    kk = 0:N÷2
    lam = sigma^2 ./ ((2pi .* kk) .^ 2 .+ tau^2) .^ alpha
    kmax === nothing || (lam = lam .* (kk .<= kmax))
    c = (randn(rng, N ÷ 2 + 1, n) .+ im .* randn(rng, N ÷ 2 + 1, n)) .* sqrt.(lam ./ 2)
    c[1, :] .= randn(rng, n) .* sqrt(lam[1])                           # 실수인 0번 모드
    c[2:end, :] .*= sqrt(2.0)
    c[end, :] .= 0
    return irfft(c, N, 1) .* N
end

"φ = a + exp(-4π²νt) cos 2πx가 φ_t = ν φ_xx를 풀므로 u = -2ν φ_x/φ는 Burgers 방정식의 정확한 해다"
function cole_hopf_exact(x, t, nu; a = 1.5)
    e = exp(-4pi^2 * nu * t)
    return 4pi * nu * e * sin(2pi * x) / (a + e * cos(2pi * x))
end

"대역 제한(삼각) 보간: 첫 축의 격자 N → Nout"
function fourier_interp(u, Nout)
    N = size(u, 1); uh = rfft(u, 1); m = min(N ÷ 2 + 1, Nout ÷ 2 + 1)
    out = zeros(eltype(uh), Nout ÷ 2 + 1, size(u)[2:end]...)
    selectdim(out, 1, 1:m) .= selectdim(uh, 1, 1:m)
    Nout > N && iseven(N) && (selectdim(out, 1, N ÷ 2 + 1) .*= 0.5)   # Nyquist 모드를 둘로 나눈다
    return irfft(out, Nout, 1) .* eltype(u)(Nout / N)
end

"Burgers 자료: 초기조건 n개의 궤적 (N, 11, n), Float32. 솔버 dt, 스냅숏은 T/10마다"
function burgers_dataset(; n, N = 512, nu = 0.005, T = 0.25, dt = 5e-4, seed = 93)
    a = grf(Xoshiro(seed), n, N)
    return Float32.(burgers_solve(a, nu, T, dt; save_every = round(Int, T / dt / 10)))
end
"KS 자료: 끌개 위의 궤적 (N, nsteps + 1, n), 저장 간격 1"
function ks_dataset(; n = 80, L = 64.0, N = 128, dt = 0.025, burn = 200.0, nsteps = 400, seed = 94)
    rng = Xoshiro(seed); x = (0:N-1) .* (L / N)
    u0 = zeros(N, n)
    for m in 1:8
        u0 .+= (rand(rng, 1, n) .- 0.5) .* cos.(2pi * m .* x ./ L .+ 2pi .* rand(rng, 1, n))
    end
    u0 = ks_solve(u0, L, burn, dt)                                     # 끌개 위로
    return ks_solve(u0, L, nsteps * 1.0, dt; save_every = round(Int, 1 / dt))
end
