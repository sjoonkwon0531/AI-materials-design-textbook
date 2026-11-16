# Lab 2.10 — spectral (FFT) differentiation vs 2nd-order central FDM for u = exp(sin x) on [0, 2π)
# (Julia port of the chapter code; FFTW.jl replaces numpy.fft)
using FFTW, Printf

u(x)  = exp(sin(x))
du(x) = cos(x) * exp(sin(x))

function spectral_deriv(N)                     # the "three-line magic"
    x = range(0, 2pi, length=N + 1)[1:N]       # = np.linspace(0, 2π, N, endpoint=False)
    k = fftfreq(N, N)                          # integer wavenumbers 0,1,…,N/2-1,-N/2,…,-1 (== np.fft.fftfreq(N, 1/N))
    dp = real.(ifft(im .* k .* fft(u.(x))))
    return maximum(abs.(dp .- du.(x)))
end
function fdm_deriv(N)                          # 2nd-order central difference, periodic
    x = range(0, 2pi, length=N + 1)[1:N]; h = 2pi / N
    uh = u.(x); dp = (circshift(uh, -1) .- circshift(uh, 1)) ./ (2h)
    return maximum(abs.(dp .- du.(x)))
end

@printf("%5s %14s %12s %11s\n", "N", "spectral_err", "FDM2_err", "FDM_order")
let prev = NaN, prevN = 0
    for N in (4, 8, 12, 16, 24, 32, 48, 64)
        s = spectral_deriv(N); f = fdm_deriv(N)
        o = isnan(prev) ? "--" : @sprintf("%.3f", log(prev / f) / log(N / prevN))
        @printf("%5d %14.3e %12.3e %11s\n", N, s, f, o)
        prev = f; prevN = N
    end
end
