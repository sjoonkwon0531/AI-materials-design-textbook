# Lab 5.5 (Julia) — 격자 KMC: 흡착원자 확산(BKL)과 ZGB CO 산화 상전이
# 파이썬 labs/53_lattice.py 대응.
#  Part A: rejection-free(BKL/체류시간) KMC로 흡착원자 확산계수 D(T), 결합 계수 장벽 E = Ea + n Eb
#  Part B: Ziff-Gulari-Barshad(1986) CO 산화 모형의 동역학적 상도표
# 1-based 인덱스: 격자 자리는 1..L, 주기 이웃은 mod1. 난수 결과는 통계적으로만 일치한다.
using Printf, Random, Statistics

const KB = 8.617333262e-5                     # eV/K
const DI = (1, -1, 0, 0); const DJ = (0, 0, 1, -1)

"""Rejection-free KMC. 원자는 빈 최근접 자리로 속도 nu exp(-(Ea + n_occ Eb)/kT)로 뛴다
(n_occ = 출발 자리의 점유된 이웃 수). 측정 구간의 언랩 위치를 균일한 물리 시간 격자에 기록하고,
시간 원점과 원자에 대해 평균한 MSD(lag)의 원점 통과 기울기로 D를 구한다. (D, 측정 구간 시간)"""
function kmc_adatoms(L, N, Ea, Eb, nu, T, n_events, n_equil, seed; nrec = 400)
    Random.seed!(seed)
    beta = 1.0 / (KB * T)
    occ = zeros(Int, L, L)                    # 0 = 빈 자리, 아니면 원자 번호(1..N)
    pos = zeros(Int, 2, N); unw = zeros(Int, 2, N)
    placed = 0
    while placed < N
        i = rand(1:L); j = rand(1:L)
        if occ[i, j] == 0
            placed += 1; occ[i, j] = placed; pos[1, placed] = i; pos[2, placed] = j
        end
    end
    krate = [nu * exp(-beta * (Ea + n * Eb)) for n in 0:4]   # krate[n_occ + 1] (BKL 부류)
    rates = zeros(4 * N)
    t = 0.0; t_meas0 = 0.0; dt_rec = 0.0
    rec = zeros(2, N, nrec); krec = 0
    for ev in 0:(n_equil+n_events-1)
        if ev == n_equil
            t_meas0 = t; unw .= 0
        end
        R = 0.0                                # 사건 목록 (교육용 O(N); 실전 코드는 부류 목록/트리)
        @inbounds for a in 1:N
            i = pos[1, a]; j = pos[2, a]; nocc = 0
            for d in 1:4
                occ[mod1(i + DI[d], L), mod1(j + DJ[d], L)] > 0 && (nocc += 1)
            end
            for d in 1:4
                r = occ[mod1(i + DI[d], L), mod1(j + DJ[d], L)] == 0 ? krate[nocc+1] : 0.0
                rates[4*(a-1)+d] = r; R += r
            end
        end
        ev == n_equil && (dt_rec = (n_events / R) / nrec)    # 현재 평균 체류시간으로 기록 간격 결정
        tau = -log(rand()) / R                                # 물리 시간 증분, 식 (5.3.5)
        if ev >= n_equil
            while krec < nrec && t + tau > t_meas0 + krec * dt_rec
                krec += 1; rec[:, :, krec] .= unw
            end
        end
        t += tau
        r = rand() * R; c = 0.0; e = 0                        # 속도에 비례해 사건 선택
        while true
            e += 1; c += rates[e]
            c >= r && break
        end
        a = (e - 1) ÷ 4 + 1; d = (e - 1) % 4 + 1
        i = pos[1, a]; j = pos[2, a]; ni = mod1(i + DI[d], L); nj = mod1(j + DJ[d], L)
        occ[i, j] = 0; occ[ni, nj] = a; pos[1, a] = ni; pos[2, a] = nj
        unw[1, a] += DI[d]; unw[2, a] += DJ[d]
    end
    nl = krec ÷ 4; lags = zeros(nl); msd = zeros(nl)          # 시간 원점 평균 MSD(lag)
    for l in 1:nl
        acc = 0.0; cnt = 0
        for o in 1:(krec-l), a in 1:N
            dx = rec[1, a, o+l] - rec[1, a, o]; dy = rec[2, a, o+l] - rec[2, a, o]
            acc += dx * dx + dy * dy; cnt += 1
        end
        lags[l] = l * dt_rec; msd[l] = acc / cnt
    end
    D = sum(lags .* msd) / sum(lags .* lags) / 4.0            # 원점 통과 최소제곱 기울기 / 4
    return D, t - t_meas0
end

"ZGB 모형. 0 빈 자리, 1 CO, 2 O. 1 MCS = L^2 흡착 시도. (<theta_CO>, <theta_O>, 자리·MCS당 CO2)"
function zgb(L, y, n_warm, n_meas, seed)
    Random.seed!(seed)
    s = zeros(Int8, L, L); nCO = 0; nO = 0; sumCO = 0.0; sumO = 0.0; nprod = 0
    for mcs in 0:(n_warm+n_meas-1)
        @inbounds for _ in 1:L*L
            i = rand(1:L); j = rand(1:L)
            if rand() < y                                    # CO 흡착 시도
                s[i, j] != 0 && continue                     # 차 있으면 아무 일도 없음
                d0 = rand(0:3); reacted = false              # 이웃 O 찾기 (무작위 시작 방향)
                for kk in 0:3
                    d = (d0 + kk) % 4 + 1
                    ii = mod1(i + DI[d], L); jj = mod1(j + DJ[d], L)
                    if s[ii, jj] == 2
                        s[ii, jj] = 0; nO -= 1; reacted = true
                        mcs >= n_warm && (nprod += 1)
                        break
                    end
                end
                reacted || (s[i, j] = 1; nCO += 1)
            else                                             # O2 해리 흡착: 무작위 이웃 쌍
                d = rand(1:4); i2 = mod1(i + DI[d], L); j2 = mod1(j + DJ[d], L)
                (s[i, j] != 0 || s[i2, j2] != 0) && continue
                s[i, j] = 2; s[i2, j2] = 2; nO += 2
                for (pi_, pj) in ((i, j), (i2, j2))           # 각 O마다 이웃 CO와 즉시 반응
                    d0 = rand(0:3)
                    for kk in 0:3
                        dd = (d0 + kk) % 4 + 1
                        ii = mod1(pi_ + DI[dd], L); jj = mod1(pj + DJ[dd], L)
                        if s[ii, jj] == 1
                            s[ii, jj] = 0; s[pi_, pj] = 0; nCO -= 1; nO -= 1
                            mcs >= n_warm && (nprod += 1)
                            break
                        end
                    end
                end
            end
        end
        if mcs >= n_warm
            sumCO += nCO; sumO += nO
        end
    end
    N2 = L * L
    return sumCO / n_meas / N2, sumO / n_meas / N2, nprod / n_meas / N2
end

function main()
    println("=== Part A: adatom diffusion, rejection-free KMC (BKL) ===")
    nu, Ea, L = 1.0e13, 0.50, 40
    Ts = [300.0, 350.0, 400.0, 500.0, 600.0, 800.0]
    println("(i) single adatom, L=40, 200 independent walkers x 20000 hops, a=1 (lattice units):")
    res1 = Float64[]
    for T in Ts
        Dw = [kmc_adatoms(L, 1, Ea, 0.0, nu, T, 20000, 0, 1000 + w; nrec = 200)[1] for w in 0:199]
        D = mean(Dw); dD = std(Dw) / sqrt(length(Dw)); Dex = nu * exp(-Ea / (KB * T))
        push!(res1, D)
        @printf("  T=%5.0f K  D_KMC=%.3e +/- %.1e  D_exact=%.3e  ratio=%.3f\n", T, D, dD, Dex, D / Dex)
    end
    sl, ic = linfit(1.0 ./ (KB .* Ts), log.(res1))
    @printf("  Arrhenius fit: Ea_fit=%.3f eV (input %.2f), prefactor=%.2e (input %.1e)\n", -sl, Ea, exp(ic), nu)
    N = 160                                                  # theta = 160/1600 = 0.10
    for Eb in (0.0, 0.10)
        @printf("(ii) lattice gas L=%d, N=%d (theta=0.10), Eb=%.2f eV, 1e5 equil + 3e5 events, 3 seeds:\n", L, N, Eb)
        Ds = Float64[]
        for T in Ts
            t0 = time(); vals = Float64[]; ttot = 0.0
            for sd in 0:2
                D, ttot = kmc_adatoms(L, N, Ea, Eb, nu, T, 300000, 100000, 100 + sd); push!(vals, D)
            end
            push!(Ds, mean(vals))
            @printf("  T=%5.0f K  D=%.3e +/- %.1e  D/D_single=%.3f  t_sim=%.2e s  (%.1f s wall)\n",
                    T, mean(vals), std(vals) / sqrt(3), mean(vals) / (nu * exp(-Ea / (KB * T))), ttot, time() - t0)
        end
        hiT = Ts .>= 500
        s_all = linfit(1.0 ./ (KB .* Ts), log.(Ds))[1]
        s_hi = linfit(1.0 ./ (KB .* Ts[hiT]), log.(Ds[hiT]))[1]
        s_lo = linfit(1.0 ./ (KB .* Ts[.!hiT]), log.(Ds[.!hiT]))[1]
        @printf("  apparent Ea (all T)=%.3f eV; T>=500 K: %.3f eV; T<500 K: %.3f eV\n", -s_all, -s_hi, -s_lo)
    end
    D, ttot = kmc_adatoms(L, N, Ea, 0.10, nu, 300.0, 300000, 100000, 100)
    @printf("  300 K, Eb=0.10: 3e5 production events cover t_sim=%.2e s = %.1e MD steps of 1 fs\n", ttot, ttot / 1e-15)

    println("=== Part B: ZGB CO oxidation, L=128, 1500 warm-up + 1500 measured MCS, seed=1 ===")
    zgb(16, 0.5, 2, 2, 0)                                    # JIT 예열
    ys = sort(unique(round.(vcat([0.30, 0.34, 0.37], 0.38:0.005:0.4001, [0.42, 0.44, 0.46, 0.48],
                                 0.50:0.005:0.5401, [0.56, 0.60]); digits = 3)))
    t0 = time(); rows = zeros(length(ys), 4)
    for (k, y) in enumerate(ys)
        tc, to, rp = zgb(128, y, 1500, 1500, 1)
        rows[k, :] = [y, tc, to, rp]
        @printf("  y=%.3f  theta_CO=%.3f  theta_O=%.3f  R_CO2=%.4f\n", y, tc, to, rp)
    end
    @printf("  wall time %.0f s\n", time() - t0)
    act = rows[rows[:, 4] .> 1e-3, :]
    @printf("  reactive window (R_CO2>1e-3): %.3f <= y <= %.3f\n", act[1, 1], act[end, 1])
    imax = argmax(rows[:, 4])
    @printf("  max R_CO2=%.4f at y=%.3f\n", rows[imax, 4], rows[imax, 1])
end

"최소제곱 직선 (np.polyfit(x, y, 1)): (기울기, 절편)"
function linfit(x, y)
    p = hcat(collect(Float64, x), ones(length(x))) \ collect(Float64, y)
    return p[1], p[2]
end

main()
