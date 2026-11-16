# lab_9_19.jl — Lab 9.19: Ch.2의 솔버를 다시 세우고, 기준선이 얼마나 믿을 만한지부터 잰다 (labs/96_ch_solver.py, 96_ch_data.py)
# (i) Lab 2.14의 설정(256², T = 400)에서 세 가지 구조 점검을 다시 한다. (ii) 64² 격자에서 궤적 248개를 만든다
# (급랭 c0 = m + 0.1·U(-1, 1), Δt = 2.5(솔버 50걸음)마다 기록, T = 50). (iii) 기준 해의 시간 수렴과 스킴의 증폭인자를 잰다.
# 자료는 models/96_ch.jls에 저장한다((64, 64, 21, 궤적) Float32). 스킴과 진단 함수는 ch2d.jl.
include("ch9_common.jl")
include("ch2d.jl")

function lab214()
    N = 256; sol = CH2D(; N, L = 4pi); c = 0.1 .* (2 .* rand(Xoshiro(1), N, N) .- 1)
    mean0 = mean(c); t = [0.0]; F = [free_energy(sol, c)[1]]; Lt = [char_length(sol, c)[1]]; drift = 0.0; t0 = time()
    for n in 1:8000
        c = advance(sol, c, 1); drift = max(drift, abs(mean(c) - mean0))
        n % 20 == 0 && (push!(t, n * sol.dt); push!(F, free_energy(sol, c)[1]); push!(Lt, char_length(sol, c)[1]))
    end
    println("Lab 2.14 re-run with the batched re-implementation (256^2, T = 400)")
    @printf("[1] MASS conservation:   max |mean(t)-mean0| = %.2e\n", drift)
    @printf("[2] FREE ENERGY:         F: %.2f -> %.2f,  F-increase steps = %d / %d\n", F[1], F[end], count(>(0), diff(F)), length(F) - 1)
    @printf("[3] COARSENING:          late-time log-log slope = %.2f\n", growth_slope(t, Lt; lo = 50.0, hi = 400.0))
    @printf("time %.1f s  (%.2f ms per step)\n", time() - t0, 1e3 * (time() - t0) / 8000)
end

function make_data(; N = 64, steps = 50, nrec = 20)
    sol = CH2D(; N); t0 = time()
    rng = Xoshiro(0); m_tr = 0.4 .* rand(rng, 160) .- 0.2; tr = trajectory(sol, random_ic(rng, 160, N, m_tr), nrec, steps)
    rng = Xoshiro(1); m_te = 0.4 .* rand(rng, 40) .- 0.2; te = trajectory(sol, random_ic(rng, 40, N, m_te), nrec, steps)
    m_ood = repeat([0.3, 0.4, 0.5]; inner = 16) .* repeat(repeat([1.0, -1.0]; inner = 8), 3)
    ood = trajectory(sol, random_ic(Xoshiro(2), 48, N, m_ood), nrec, steps); el = time() - t0
    serialize(joinpath(CH9_OUT, "96_ch.jls"), (train = Float32.(tr), test = Float32.(te), ood = Float32.(ood), m_test = m_te, m_ood = m_ood, seconds = el))
    @printf("grid %d^2, L = 2 pi, dx = %.4f, eps = %.1f, dt = %.2f, S = %.1f; record every %d steps (Dt = %.1f)\n", N, sol.dx, sol.eps, sol.dt, sol.S, steps, steps * sol.dt)
    @printf("all 248 trajectories (%d solver steps): %.1f s = %.3f ms per solver step per sample\n", 248 * nrec * steps, el, 1e3 * el / 248 / (nrec * steps))
    t = collect(0:nrec) .* (steps * sol.dt)
    F = reduce(hcat, [free_energy(sol, te[:, :, i, :]) for i in 1:nrec+1])'            # (시각, 궤적)
    Lt = reduce(hcat, [char_length(sol, te[:, :, i, :]) for i in 1:nrec+1])'
    slopes = [growth_slope(t, Lt[:, b]) for b in 1:40]
    @printf("[1] mass: max |mean(t) - mean(0)| over 40 test trajectories = %.2e\n", maximum(mass_drift(te)))
    @printf("[2] free energy: mean F %.2f -> %.2f; increases between records: %d / %d\n", mean(F[1, :]), mean(F[end, :]), count(>(0), diff(F; dims = 1)), 40 * nrec)
    @printf("[3] L(t): %.2f (t=2.5) -> %.2f (t=50); slope of mean L on 5 <= t <= 50: %.3f; per-trajectory slopes %.3f +/- %.3f (s.d.)\n",
            mean(Lt[2, :]), mean(Lt[end, :]), growth_slope(t, vec(mean(Lt; dims = 2))), mean(slopes), std(slopes; corrected = false))
    rel(a, b) = mean(sqrt.(sum(abs2, a .- b; dims = (1, 2)) ./ sum(abs2, b; dims = (1, 2))))
    println("persistence error of one big step starting at t = 0, 2.5, 5, 10, 25, 47.5: ", join([@sprintf("%.3f", rel(te[:, :, i, :], te[:, :, i+1, :])) for i in (1, 2, 3, 5, 11, 20)], " "))

    # (i) 기준 해는 시간에 대해 얼마나 수렴했는가: 형성된 미세구조 c(5), c(25)에서 큰 한 걸음(Δt = 2.5), dt = 0.0025와 비교
    println("one big step (Dt = 2.5) from the formed microstructure c(5) and c(25), 40 test trajectories, vs dt = 0.0025:")
    fine = CH2D(; N, dt = 0.0025); r5 = advance(fine, te[:, :, 3, :], 1000); r25 = advance(fine, te[:, :, 11, :], 1000)
    for dt in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 2.5)
        s2 = CH2D(; N, dt); n = round(Int, 2.5 / dt)
        @printf("   dt = %-6g: rel. L2 from c(5): %.4f   from c(25): %.4f\n", dt, rel(advance(s2, te[:, :, 3, :], n), r5), rel(advance(s2, te[:, :, 11, :], n), r25))
    end
    # (ii) 균일 상태 c = m 둘레의 von Neumann 증폭인자: 스킴(dt = 0.05) 대 정확한 exp(σ)
    println("\namplification of the fastest mode per unit time, scheme (dt = 0.05) vs exact exp(sigma):")
    k = collect(1.0:32.0); DT, S, EPS = sol.dt, sol.S, sol.eps
    for m in (0.0, 0.2, 0.3, 0.4, 0.5)
        a = 1 - 3m^2
        g = (1 .+ DT .* k .^ 2 .* (S + a)) ./ (1 .+ DT * S .* k .^ 2 .+ DT * EPS^2 .* k .^ 4)
        sig = k .^ 2 .* (a .- EPS^2 .* k .^ 2)
        onset = log(1.0 / 0.0577) / (log(maximum(g)) / DT)                             # rms 잡음 0.0577이 O(1)에 이르는 시간
        @printf("   m = %.1f: scheme  max g^(1/dt) = %8.2f at k = %.0f;  PDE  exp(sigma_max) = %10.3g at k = %.0f;  time for 0.058 -> 1 with the scheme: %.2f\n",
                m, maximum(g)^(1 / DT), k[argmax(g)], exp(maximum(sig)), k[argmax(sig)], onset)
    end
end

lab214(); make_data()
