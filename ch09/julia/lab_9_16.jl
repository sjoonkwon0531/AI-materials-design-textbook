# lab_9_16.jl — Lab 9.16: 의사 스펙트럴 Navier–Stokes 솔버, 검증하고 자료를 만든다 (labs/95_ns_solver.py, 95_ns_data.py)
# 솔버는 ns2d.jl. [1] Taylor–Green 와류의 점성 감쇠(정확해), [2] 시간 수렴, [3] 공간 수렴, [4] 2/3 규칙의 효과,
# 그리고 자료: 초기조건 240개(Gauss 무작위장), ν = 1e-3, 64², dt = 1e-2, T = 20까지 1.0마다 기록 → models/95_ns.jls.
include("ch9_common.jl")
include("ns2d.jl")

function selftest()
    N, nu = 64, 1e-2; x = (0:N-1) ./ N; k = 2pi
    w0 = reshape([2k * cos(k * xx) * cos(k * yy) for yy in x, xx in x], N, N, 1)     # w = 2k cos kx cos ky는 exp(-2νk²t)로 감쇠
    w1 = advance(NS2D(N; nu, dt = 1e-3, force = false), w0, 1000)
    @printf("[1] Taylor-Green, N=64, nu=1e-2, t=1: rel. L2 error = %.2e\n", rel2d(w1, w0 .* exp(-2 * nu * k^2))[1])
    w128 = grf2d(Xoshiro(0), 4, 128); w64 = spectral_resample(w128, 64)
    @printf("[2] GRF sample (4 fields): std(w0) = %.3f, max|w0| = %.3f\n", std(w128; corrected = false), maximum(abs, w128))
    ref_t = advance(NS2D(64; dt = 2.5e-4), w64, 16000); prev = NaN                   # t = 4
    for dt in (2e-2, 1e-2, 5e-3, 2.5e-3)
        e = mean(rel2d(advance(NS2D(64; dt), w64, round(Int, 4 / dt)), ref_t))
        @printf("    N=64  dt=%.1e: error vs dt=2.5e-4 at t=4 = %.2e%s\n", dt, e, isnan(prev) ? "" : @sprintf("   (ratio %.2f)", prev / e)); prev = e
    end
    ref_x = spectral_resample(advance(NS2D(128; dt = 1e-3), w128, 4000), 64)
    for (M, dt) in ((64, 2.5e-3), (48, 5e-3), (32, 5e-3), (24, 1e-2), (16, 1e-2))
        w = advance(NS2D(M; dt), spectral_resample(w128, M), round(Int, 4 / dt))
        @printf("[3] N=%-3d dt=%.1e: error vs N=128 at t=4 (on the 64^2 grid) = %.2e\n", M, dt, mean(rel2d(spectral_resample(w, 64), ref_x)))
    end
    ref20 = spectral_resample(advance(NS2D(128; dt = 2.5e-3), w128, 8000), 64)       # [4] t = 20, 2/3 규칙을 끄면
    for (M, dt, n) in ((64, 1e-2, 2000), (16, 2e-2, 1000))
        wM = spectral_resample(w128, M)
        ea = mean(rel2d(spectral_resample(advance(NS2D(M; dt), wM, n), 64), ref20))
        eb = mean(rel2d(spectral_resample(advance(NS2D(M; dt, dealias = false), wM, n), 64), ref20))
        @printf("[4] N=%d, t=20: error with 2/3 dealiasing = %.2e, without = %.2e\n", M, ea, eb)
    end
    s = NS2D(64); @printf("    2/3 mask at N = 64: keeps |k| <= %d in each direction, %d of %d stored modes\n", floor(Int, 2 / 3 * 32), count(>(0), s.mask), length(s.mask))
end

function make_data(; N = 64, nu = 1e-3, dt = 1e-2, T = 20, ntraj = 240, batch = 40, seed = 0)
    rng = Xoshiro(seed); solver = NS2D(N; nu, dt)
    w = zeros(Float32, N, N, T + 1, ntraj); w0 = zeros(N, N, ntraj); t0 = time()
    for b in 1:batch:ntraj
        ii = b:b+batch-1; w0[:, :, ii] = grf2d(rng, batch, N)
        w[:, :, :, ii] = trajectory(solver, w0[:, :, ii], T, round(Int, 1 / dt))
    end
    el = time() - t0
    serialize(joinpath(CH9_OUT, "95_ns.jls"), (w = w, w0 = w0, seconds = el))
    @printf("generated %d trajectories x %d unit-time steps (%d solver steps) in %.1f s = %.1f ms per sample per unit time\n",
            ntraj, T, ntraj * T * round(Int, 1 / dt), el, 1e3 * el / ntraj / T)
    println("std of w at t = 0, 5, 10, 20: ", join([@sprintf("%.3f", std(w[:, :, i, :]; corrected = false)) for i in (1, 6, 11, 21)], " "))
    pers = [mean(rel2d(w[:, :, i, :], w[:, :, i+1, :])) for i in (1, 6, 11, 20)]
    println("persistence error ||w(t+1)-w(t)||/||w(t+1)|| at t = 0, 5, 10, 19: ", join([@sprintf("%.3f", e) for e in pers], " "))
end

selftest(); make_data()
