# Lab 5.2 (Julia) — 임계 감속을 재고, Wolff로 푼다
# 파이썬 labs/52_tau.py 대응: 2D Ising, beta_c = ln(1+sqrt2)/2, 무작위 자리 Metropolis 대 Wolff.
# 시간 단위는 둘 다 sweep (Wolff: 클러스터 수 x <C>/L^2). 난수가 numpy와 다르므로 tau 값은
# 통계적으로만 일치한다(작은 L에서 수 %, Metropolis L=64에서 20 % 이상 흔들림은 파이썬과 같음).
using Printf, Random, Statistics, FFTW
include("isingkern.jl")

function main()
    bc = 0.5 * log(1 + sqrt(2)); t0 = time()
    metro_runs = [(8, 600000), (16, 400000), (32, 500000), (64, 400000)]
    wolff_runs = [(8, 200000), (16, 200000), (32, 150000), (64, 80000), (128, 40000)]
    run_metro2d(8, bc, 10, 10, 1); run_wolff2d(8, bc, 10, 10, 1)          # JIT 예열 (CPU 시간에서 제외)

    println("=== Metropolis (random-site), T = T_c, 2000 eq sweeps; tau in sweeps ===")
    println("   L   sweeps    tau_int(m^2)        tau_int(E)       CPU s")
    res_m = Dict{Int,NamedTuple}(); blk = Float64[]; m2_32 = Float64[]
    for (L, n) in metro_runs
        t = time()
        E, M = run_metro2d(L, bc, 2000, n, 100 + L)
        m2 = (M ./ L^2) .^ 2
        tm = tau_int(m2); te = tau_int(E)
        cpu = time() - t
        res_m[L] = (tm = tm[1], tm_err = tm[2], te = te[1], te_err = te[2], cpu = cpu, n = n)
        if L == 32
            blk = blocking(m2); m2_32 = m2
        end
        @printf("  %3d  %7d   %7.1f +/- %5.1f   %6.1f +/- %4.1f   %6.1f\n", L, n, tm[1], tm[2], te[1], te[2], cpu)
    end

    println("=== Wolff, T = T_c, 1000 eq clusters; tau converted to sweeps = tau_cluster * <C>/L^2 ===")
    println("   L  clusters  <C>/L^2   tau_int(m^2)      tau_int(E)      CPU s")
    res_w = Dict{Int,NamedTuple}()
    for (L, n) in wolff_runs
        t = time()
        E, M, C = run_wolff2d(L, bc, 1000, n, 200 + L)
        f = mean(C) / L^2
        tm = tau_int((M ./ L^2) .^ 2); te = tau_int(E)
        cpu = time() - t
        res_w[L] = (f = f, C = mean(C), tm = tm[1] * f, tm_err = tm[2] * f, te = te[1] * f, te_err = te[2] * f,
                    cpu = cpu, n = n)
        @printf("  %3d  %7d   %.4f   %6.3f +/- %5.3f   %6.3f +/- %5.3f  %6.1f\n",
                L, n, f, tm[1] * f, tm[2] * f, te[1] * f, te[2] * f, cpu)
    end

    # 로그-로그 직선 맞춤으로 동역학 지수
    Lm = [L for (L, _) in metro_runs]; Lw = [L for (L, _) in wolff_runs]
    zm = linfit(log.(Lm), log.([res_m[L].tm for L in Lm]))[1]
    zm2 = linfit(log.(Lm[2:end]), log.([res_m[L].tm for L in Lm[2:end]]))[1]
    zw_m = linfit(log.(Lw), log.([res_w[L].tm for L in Lw]))[1]
    zw_e = linfit(log.(Lw), log.([res_w[L].te for L in Lw]))[1]
    gC = linfit(log.(Lw), log.([res_w[L].C for L in Lw]))[1]
    println("=== fits ===")
    @printf("  Metropolis z (m^2): L=8-64 -> %.2f;  L=16-64 -> %.2f\n", zm, zm2)
    @printf("  Wolff z (m^2) = %.2f,  Wolff z (E) = %.2f   (L = 8-128)\n", zw_m, zw_e)
    @printf("  Wolff <C> ~ L^%.3f   (gamma/nu = 1.75)\n", gC)

    # Metropolis L = 32의 블로킹 분석
    naive = blk[1]
    plateau = length(blk) > 4 ? maximum(blk[1:end-3]) : blk[end]
    println("=== blocking analysis, Metropolis L=32, observable m^2 ===")
    println("  level  blocks  std.err")
    for (k0, e) in enumerate(blk)
        k = k0 - 1                                         # 파이썬의 0-based 수준 번호
        (k % 2 == 0 || k == length(blk) - 1) && @printf("  %5d  %6d  %.2e\n", k, res_m[32].n ÷ 2^k, e)
    end
    @printf("  <m^2> = %.4f;  naive err = %.2e;  plateau err = %.2e\n", mean(m2_32), naive, plateau)
    @printf("  (plateau/naive)^2 = %.0f   vs   2*tau_int = %.0f\n", (plateau / naive)^2, 2 * res_m[32].tm)

    # L = 64에서 독립 표본 하나(2 tau_int)의 CPU 비용
    L = 64
    cm = res_m[L].cpu / res_m[L].n * 2 * res_m[L].tm
    cw = res_w[L].cpu / (res_w[L].n * res_w[L].f) * 2 * res_w[L].tm
    @printf("=== CPU seconds per independent sample (2 tau_int) at L=%d: Metropolis %.1f ms, Wolff %.2f ms, ratio %.0f ===\n",
            L, cm * 1e3, cw * 1e3, cm / cw)
    @printf("total %.0f s\n", time() - t0)
end

main()
