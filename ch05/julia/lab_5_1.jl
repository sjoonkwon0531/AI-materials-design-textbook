# Lab 5.1 (Julia) — Lennard-Jones 유체의 NVT·NPT Monte Carlo (+ Langevin MD, 벽 경계)
# 파이썬 labs/52_ljmc.py와 같은 알고리즘·매개변수·출력 형식.
# 환산 LJ 단위(epsilon = sigma = m = k_B = 1), N = 256, r_c = 2.5 + 꼬리 보정.
# 배열 배치: column-major이므로 좌표를 3 x N(열 = 입자)으로 둔다.
# 난수: Julia RNG는 numpy와 다르므로 모든 숫자는 통계적으로만(오차막대 안에서) 일치한다.
# 기준 EOS(Thol et al. 2016)는 파이썬 판이 NIST teqp로 계산한 값을 상수로 옮겨 적었다.
using Printf, Random, Statistics

const RC = 2.5
const RC2 = RC * RC

# ---------------------------------------------------------------- 기준 EOS 값 (teqp, T* = 2.0)
const EOS_U = Dict(0.1 => -0.6674, 0.3 => -1.9302, 0.5 => -3.1525, 0.7 => -4.3024, 0.9 => -5.0263)
const EOS_P = Dict(0.1 => 0.1778, 0.3 => 0.4874, 0.5 => 1.0752, 0.7 => 3.0320, 0.9 => 9.0898)
const EOS_RHO = Dict(1.0 => 0.4840, 3.0 => 0.6981)

# ---------------------------------------------------------------- 유틸리티
"Flyvbjerg-Petersen 블로킹: (수준별 오차, 평탄값 = 블록이 충분한 수준들의 최댓값)"
function blocking_error(x)
    x = collect(Float64, x); errs = Float64[]
    while length(x) >= 32
        n = length(x)
        push!(errs, sqrt(var(x) / n))            # var는 기본이 ddof = 1 (numpy ddof=1과 같음)
        m = 2 * (n ÷ 2)
        x = 0.5 .* (x[1:2:m] .+ x[2:2:m])         # 파이썬 x[0:m:2], x[1:m:2]
    end
    return errs, maximum(errs[1:max(1, length(errs) - 2)])
end

u_tail(rho) = 8.0 / 3.0 * pi * rho * (1.0 / 3.0 * RC^-9 - RC^-3)
p_tail(rho) = 16.0 / 3.0 * pi * rho^2 * (2.0 / 3.0 * RC^-9 - RC^-3)

function fcc_lattice(N, L)
    nc = round(Int, (N / 4)^(1 / 3)); @assert 4 * nc^3 == N
    a = L / nc
    basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    pos = zeros(3, N); k = 0
    for i in 0:nc-1, j in 0:nc-1, l in 0:nc-1, b in basis      # 파이썬과 같은 순서 (i 바깥, b 안쪽)
        k += 1
        pos[1, k] = (i + b[1] + 0.25) * a
        pos[2, k] = (j + b[2] + 0.25) * a
        pos[3, k] = (l + b[3] + 0.25) * a
    end
    return pos
end

# ---------------------------------------------------------------- 에너지와 virial
"LJ 쌍 에너지와 virial r.F (r2 < RC2). shift = true이면 절단-이동 퍼텐셜"
@inline function pair_ev(r2, shift)
    inv6 = 1.0 / (r2 * r2 * r2)
    e = 4.0 * inv6 * (inv6 - 1.0)
    if shift
        ic6 = 1.0 / (RC2 * RC2 * RC2)
        e -= 4.0 * ic6 * (ic6 - 1.0)
    end
    return e, 24.0 * inv6 * (2.0 * inv6 - 1.0)
end

"입자 i를 (rx, ry, rz)에 두었을 때 나머지와의 에너지·virial: O(N)"
function particle_ev(pos, i, rx, ry, rz, L, Lz, walls, shift)
    e = 0.0; w = 0.0
    @inbounds for j in 1:size(pos, 2)
        j == i && continue
        dx = rx - pos[1, j]; dy = ry - pos[2, j]; dz = rz - pos[3, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)   # round: ties-to-even (np.round과 같음)
        walls || (dz -= Lz * round(dz / Lz))
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < RC2
            a, b = pair_ev(r2, shift); e += a; w += b
        end
    end
    return e, w
end

function total_ev(pos, L, Lz, walls, shift)
    N = size(pos, 2); e = 0.0; w = 0.0
    @inbounds for i in 1:N-1, j in i+1:N
        dx = pos[1, i] - pos[1, j]; dy = pos[2, i] - pos[2, j]; dz = pos[3, i] - pos[3, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        walls || (dz -= Lz * round(dz / Lz))
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < RC2
            a, b = pair_ev(r2, shift); e += a; w += b
        end
    end
    return e, w
end

"N번의 단일 입자 Metropolis 변위 시도. (E, W, 수용 수)를 돌려준다."
function nvt_sweep!(pos, L, Lz, beta, delta, E, W, walls, shift)
    N = size(pos, 2); nacc = 0
    @inbounds for _ in 1:N
        i = rand(1:N)
        e_old, w_old = particle_ev(pos, i, pos[1, i], pos[2, i], pos[3, i], L, Lz, walls, shift)
        rx = pos[1, i] + delta * (2.0 * rand() - 1.0)               # 대칭 제안
        ry = pos[2, i] + delta * (2.0 * rand() - 1.0)
        rz = pos[3, i] + delta * (2.0 * rand() - 1.0)
        if walls
            (rz < 0.0 || rz > Lz) && continue                       # 단단한 벽: 밖으로 나가면 거절
            rx -= L * floor(rx / L); ry -= L * floor(ry / L)
        else
            rx -= L * floor(rx / L); ry -= L * floor(ry / L); rz -= Lz * floor(rz / Lz)
        end
        e_new, w_new = particle_ev(pos, i, rx, ry, rz, L, Lz, walls, shift)
        dE = e_new - e_old
        if dE <= 0.0 || rand() < exp(-beta * dE)                     # Metropolis
            pos[1, i] = rx; pos[2, i] = ry; pos[3, i] = rz
            E += dE; W += w_new - w_old; nacc += 1
        end
    end
    return E, W, nacc
end

# ---------------------------------------------------------------- Part A: NVT
function run_nvt(rho, T; N = 256, n_eq = 3000, n_prod = 20000, seed = 1, shift = false)
    Random.seed!(seed)
    L = (N / rho)^(1 / 3); pos = fcc_lattice(N, L); beta = 1.0 / T
    E, W = total_ev(pos, L, L, false, shift)
    delta = 0.1
    for s in 1:n_eq                                   # 평형화 구간에서만 수용률 40 % 목표로 조정
        E, W, na = nvt_sweep!(pos, L, L, beta, delta, E, W, false, shift)
        if s % 50 == 0
            delta *= na / N > 0.40 ? 1.05 : 0.95
            delta = min(delta, L / 2)
        end
    end
    E, W = total_ev(pos, L, L, false, shift)
    V = L^3; us = zeros(n_prod); ps = zeros(n_prod); nacc_tot = 0
    for s in 1:n_prod
        E, W, na = nvt_sweep!(pos, L, L, beta, delta, E, W, false, shift)
        nacc_tot += na
        us[s] = E / N; ps[s] = rho * T + W / (3 * V)
    end
    E2, _ = total_ev(pos, L, L, false, shift)
    if !shift
        us .+= u_tail(rho); ps .+= p_tail(rho)
    end
    return (rho = rho, delta = delta, acc = nacc_tot / (N * n_prod), drift = abs(E2 - E),
            u = mean(us), u_err = blocking_error(us)[2], p = mean(ps), p_err = blocking_error(ps)[2])
end

# ---------------------------------------------------------------- Part B: NPT (ln V 이동)
"jac = 1: 올바른 (N+1) ln(V'/V) 항, jac = 0: 그 항을 통째로 뺀 '틀린' 버전"
function run_npt(P, T; N = 256, rho0 = 0.5, n_eq = 4000, n_prod = 20000, seed = 2, jac = 1)
    Random.seed!(seed)
    L = (N / rho0)^(1 / 3); pos = fcc_lattice(N, L); beta = 1.0 / T
    E, W = total_ev(pos, L, L, false, false)
    delta, dlnV = 0.1, 0.01
    nV_acc = 0; rhos = zeros(n_prod)
    for s in 0:n_eq+n_prod-1
        E, W, na = nvt_sweep!(pos, L, L, beta, delta, E, W, false, false)
        V = L^3
        Etot_old = E + N * u_tail(N / V)                        # 꼬리 보정은 V에 의존하므로 포함
        Vn = exp(log(V) + dlnV * (2 * rand() - 1)); Ln = Vn^(1 / 3)
        sfac = Ln / L
        pos .*= sfac                                             # s = r/L 고정: 모든 좌표를 비례 확대
        En, Wn = total_ev(pos, Ln, Ln, false, false)
        Etot_new = En + N * u_tail(N / Vn)
        arg = -beta * (Etot_new - Etot_old + P * (Vn - V)) + jac * (N + 1) * log(Vn / V)
        if arg >= 0 || rand() < exp(arg)
            L, E, W = Ln, En, Wn; nV_acc += 1; accV = true
        else
            pos ./= sfac; accV = false                           # 거절: 원래 좌표로 복원
        end
        s < n_eq && (s + 1) % 50 == 0 && (delta *= na / N > 0.40 ? 1.05 : 0.95)
        s < n_eq && (dlnV *= accV ? 1.02 : 0.99)                # 평형화 구간에서만 조정
        s == n_eq - 1 && (nV_acc = 0)
        s >= n_eq && (rhos[s-n_eq+1] = N / L^3)
    end
    return (P = P, rho = mean(rhos), rho_err = blocking_error(rhos)[2], accV = nV_acc / n_prod, dlnV = dlnV)
end

# ---------------------------------------------------------------- Part C: Langevin BAOAB MD
function forces_ts(pos, L)
    N = size(pos, 2); F = zeros(3, N); E = 0.0
    ic6 = 1.0 / (RC2 * RC2 * RC2); eshift = 4.0 * ic6 * (ic6 - 1.0)
    @inbounds for i in 1:N-1, j in i+1:N
        d1 = pos[1, i] - pos[1, j]; d2 = pos[2, i] - pos[2, j]; d3 = pos[3, i] - pos[3, j]
        d1 -= L * round(d1 / L); d2 -= L * round(d2 / L); d3 -= L * round(d3 / L)
        r2 = d1 * d1 + d2 * d2 + d3 * d3
        if r2 < RC2
            inv2 = 1.0 / r2; inv6 = inv2^3
            E += 4.0 * inv6 * (inv6 - 1.0) - eshift
            fs = (48.0 * inv6 * inv6 - 24.0 * inv6) * inv2
            F[1, i] += fs * d1; F[2, i] += fs * d2; F[3, i] += fs * d3
            F[1, j] -= fs * d1; F[2, j] -= fs * d2; F[3, j] -= fs * d3
        end
    end
    return F, E
end

"BAOAB (Leimkuhler-Matthews). every step마다 U/N과 운동에너지를 기록"
function baoab!(pos, vel, L, T, gamma, dt, nsteps, every)
    N = size(pos, 2); F, E = forces_ts(pos, L)
    c1 = exp(-gamma * dt); c2 = sqrt((1.0 - c1 * c1) * T)        # 요동-소산: c2가 c1과 T로 묶임
    out = zeros(nsteps ÷ every); Ks = zeros(nsteps ÷ every); k = 0
    for s in 1:nsteps
        vel .+= 0.5 * dt .* F                                     # B
        pos .+= 0.5 * dt .* vel                                   # A
        vel .= c1 .* vel .+ c2 .* randn(3, N)                     # O (정확한 OU 해)
        pos .+= 0.5 * dt .* vel                                   # A
        pos .-= L .* floor.(pos ./ L)
        F, E = forces_ts(pos, L)
        vel .+= 0.5 * dt .* F                                     # B
        if s % every == 0
            k += 1; out[k] = E / N; Ks[k] = 0.5 * sum(abs2, vel)
        end
    end
    return out, Ks
end

function run_langevin(rho, T; N = 256, dt = 0.005, gamma = 1.0, n_eq = 4000, n_prod = 40000, seed = 3)
    Random.seed!(seed)
    L = (N / rho)^(1 / 3); pos = fcc_lattice(N, L)
    vel = sqrt(T) .* randn(3, N)
    baoab!(pos, vel, L, T, gamma, dt, n_eq, 10)
    u, K = baoab!(pos, vel, L, T, gamma, dt, n_prod, 10)
    Tkin = 2 .* K ./ (3 * N)
    return (u = mean(u), u_err = blocking_error(u)[2], Tkin = mean(Tkin), Tkin_err = blocking_error(Tkin)[2])
end

# ---------------------------------------------------------------- Part D: z 방향 단단한 벽
function run_walls(rho, T; N = 256, n_eq = 3000, n_prod = 10000, seed = 4, nbins = 120)
    Random.seed!(seed); beta = 1.0 / T
    L = (N / (2 * rho))^(1 / 3); Lz = 2 * L                     # Lx = Ly = L, Lz = 2L
    pos = fcc_lattice(N, 1.0)                                   # 단위 정육면체 fcc를 비등방 확대
    pos[1, :] .*= L; pos[2, :] .*= L; pos[3, :] .*= Lz
    E, W = total_ev(pos, L, Lz, true, false)
    delta = 0.1
    for s in 1:n_eq
        E, W, na = nvt_sweep!(pos, L, Lz, beta, delta, E, W, true, false)
        s % 50 == 0 && (delta *= na / N > 0.40 ? 1.05 : 0.95)
    end
    hist = zeros(nbins)
    for s in 0:n_prod-1
        E, W, na = nvt_sweep!(pos, L, Lz, beta, delta, E, W, true, false)
        if s % 5 == 0                                            # np.histogram(range=(0,Lz)): 오른쪽 끝 포함
            for i in 1:N
                b = min(floor(Int, pos[3, i] / Lz * nbins) + 1, nbins)
                hist[b] += 1
            end
        end
    end
    nsamp = (n_prod + 4) ÷ 5; dz = Lz / nbins
    prof = hist ./ nsamp ./ (L * L * dz)
    zc = ((1:nbins) .- 0.5) .* dz
    return zc, prof, L, Lz
end

# ---------------------------------------------------------------- main
function main()
t0 = time(); T = 2.0
println("=== Part A: NVT MC, N=256, T*=2.0, r_c=2.5 + tail corrections, 3000 eq + 10000 prod sweeps ===")
println("  rho    delta  acc   U/N (MC)          U/N (EOS)   P (MC)            P (EOS)")
drifts = Float64[]
for rho in (0.1, 0.3, 0.5, 0.7, 0.9)
    r = run_nvt(rho, T; n_prod = 10000)
    push!(drifts, r.drift)
    @printf("  %.1f   %.3f  %.2f  %+.4f +/- %.4f  %+.4f     %.4f +/- %.4f   %.4f\n",
            rho, r.delta, r.acc, r.u, r.u_err, EOS_U[rho], r.p, r.p_err, EOS_P[rho])
end
@printf("  (max |E_incremental - E_recomputed| at end = %.1e)\n", maximum(drifts))
println("  finite-size check at rho=0.3 (2000 eq + 8000 prod sweeps, seed 11):")
for Nfs in (108, 256, 500)
    r = run_nvt(0.3, T; N = Nfs, n_eq = 2000, n_prod = 8000, seed = 11)
    @printf("    N=%4d: U/N = %+.4f +/- %.4f   P = %.4f +/- %.4f\n", Nfs, r.u, r.u_err, r.p, r.p_err)
end
@printf("  Part A time %.1f s\n", time() - t0)

t1 = time()
println("=== Part B: NPT MC, N=256, T*=2.0, one ln V trial per sweep, 3000 eq + 10000 prod sweeps ===")
for P in (1.0, 3.0)
    r = run_npt(P, T; n_eq = 3000, n_prod = 10000)
    @printf("  P=%.1f: <rho> = %.4f +/- %.4f   EOS rho = %.4f   acc(lnV) = %.2f  dlnV = %.4f\n",
            P, r.rho, r.rho_err, EOS_RHO[P], r.accV, r.dlnV)
end
rw = run_npt(1.0, T; jac = 0, seed = 2, n_eq = 3000, n_prod = 10000)
@printf("  P=1.0 WITHOUT the (N+1)ln(V'/V) term: <rho> = %.4f +/- %.4f  (wrong)\n", rw.rho, rw.rho_err)
@printf("  Part B time %.1f s\n", time() - t1)

t2 = time()
println("=== Part C: truncated-and-shifted LJ, T*=2.0, rho=0.6, N=256: Metropolis MC vs Langevin BAOAB MD ===")
rmc = run_nvt(0.6, T; shift = true, seed = 5, n_prod = 10000)
@printf("  MC  : <U_ts>/N = %+.4f +/- %.4f   (10000 sweeps)\n", rmc.u, rmc.u_err)
dt = 0.005
rmd = run_langevin(0.6, T; dt = dt, n_prod = round(Int, 150 / dt), seed = 6)
@printf("  MD  : <U_ts>/N = %+.4f +/- %.4f   kinetic T = %.4f +/- %.4f   (BAOAB, gamma=1, dt=%g, t_prod=150)\n",
        rmd.u, rmd.u_err, rmd.Tkin, rmd.Tkin_err, dt)
@printf("  Part C time %.1f s\n", time() - t2)

t3 = time()
zc, prof, Lw, Lzw = run_walls(0.8, T; n_prod = 6000)
mid = (zc .> 0.35 * Lzw) .& (zc .< 0.65 * Lzw)
println("=== Part D: hard walls in z (reflecting: moves out of the box rejected), periodic in x,y; rho=0.8, T*=2.0 ===")
@printf("  box %.3f x %.3f x %.3f; mid-slab density = %.3f; first contact peak = %.3f at z = %.2f\n",
        Lw, Lw, Lzw, mean(prof[mid]), maximum(prof), zc[argmax(prof)])
@printf("  Part D time %.1f s;  total %.1f s\n", time() - t3, time() - t0)
end

main()
