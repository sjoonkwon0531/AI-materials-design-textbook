# Lab 2.14 — 2D Cahn–Hilliard spinodal decomposition, stabilized semi-implicit Fourier spectral scheme
# (Julia port of lab2_14.py; N = 256, L = 4π, epsilon = 0.1, dt = 0.05, S = 2, 8000 steps -> T = 400)
# RNG: Julia's Xoshiro(1) is NOT numpy's default_rng(1) -> the noise field differs, so F(t), L(t)
# and the coarsening exponent agree only statistically; mass conservation and the monotone F are structural.
using FFTW, Random, Statistics, LinearAlgebra, Printf

N = 256; L = 4pi; epsilon = 0.10; dt = 0.05; S = 2.0; NSTEP = 8000
dx = L / N
k1 = 2pi .* fftfreq(N, N / L)                       # == np.fft.fftfreq(N, d=L/N) * 2π
KX = [k1[i] for i in 1:N, j in 1:N]; KY = [k1[j] for i in 1:N, j in 1:N]   # meshgrid(indexing='ij')
K2 = KX .^ 2 .+ KY .^ 2; K4 = K2 .^ 2
denom = 1 .+ dt * S .* K2 .+ dt * epsilon^2 .* K4; gain = 1 .+ dt * S .* K2
kmag = sqrt.(K2)

function energy(c)                                   # F[c] = ∫ (c^2-1)^2/4 + eps^2/2 |∇c|^2
    ch = fft(c)
    cx = real.(ifft(im .* KX .* ch)); cy = real.(ifft(im .* KY .* ch))
    return sum(0.25 .* (c .^ 2 .- 1) .^ 2 .+ 0.5 * epsilon^2 .* (cx .^ 2 .+ cy .^ 2)) * dx^2
end
function charlength(c)                               # L(t) = 2π / <k>_{S(k)}
    Sk = abs2.(fft(c .- mean(c))); Sk[1, 1] = 0
    return 2pi * sum(Sk) / sum(kmag .* Sk)
end

c = 0.1 .* (2 .* rand(Xoshiro(1), N, N) .- 1)      # quench: small noise in [-0.1, 0.1]
mean0 = mean(c); chat = fft(c)
t = [0.0]; F = [energy(c)]; Lt = [charlength(c)]; Mmax = 0.0
for n in 1:NSTEP
    global chat = (gain .* chat .- dt .* K2 .* fft(c .^ 3 .- c)) ./ denom   # Box 2.22, one line
    global c = real.(ifft(chat))
    global Mmax = max(Mmax, abs(mean(c) - mean0))    # mass drift checked every step
    if n % 20 == 0
        push!(t, n * dt); push!(F, energy(c)); push!(Lt, charlength(c))
    end
end
dF = diff(F)
@printf("[1] MASS conservation: max|mean(t)-mean0| = %.2e   (machine precision)\n", Mmax)
@printf("[2] FREE ENERGY: F: %.3g -> %.3g,  F-increase steps = %d / %d\n", F[1], F[end], count(>(0), dF), length(dF))
m = (t .>= 50) .& (t .<= 400)                        # log-log fit of L(t) on 50 <= t <= 400
X = [ones(count(m)) log.(t[m])]
slope = (X \ log.(Lt[m]))[2]                         # least squares == np.polyfit(..., 1)[0]
@printf("[3] COARSENING: late-time log-log slope = %.2f   (LSW theory: 1/3 = 0.333)\n", slope)
