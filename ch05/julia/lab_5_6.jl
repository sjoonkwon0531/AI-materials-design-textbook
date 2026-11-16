# Lab 5.6 (Julia) — Iterative Boltzmann Inversion: 이합체 유체를 단일 비드로
# 파이썬 labs/54_ibi.py 대응 (Reith, Pütz, Müller-Plathe 2003).
#  원자 모형: 2D LJ 이합체(결합 k=500, r0=0.8; 분자간 LJ, rc=2.5), Langevin BAOAB, T=1
#  CG 모형  : 이합체 하나 = 비드 하나(질량 2), 표 형태 쌍 퍼텐셜(격자 0.01, 선형 보간, rc=3.5)
#  단계     : (1) 원자 MD -> 목표 g(r), 압력  (2) PMF 초기 추정  (3) IBI  (4) 압력 보정 IBI
#             (5) 다른 밀도로의 전이성  (6) 동역학(약한 마찰)
# 좌표는 2 x n (열 = 입자). 원자 2m-1, 2m (1-based)이 분자 m을 이룬다.
# 난수 결과(g RMSE, 압력, D)는 통계적으로만 일치한다. finalize/smooth/gradient 같은 결정론적
# 후처리는 lab_5_6_xcheck.py에서 numpy 원본과 대조했다.
using Printf, Random, Statistics

const T = 1.0
const NMOL = 200
const RHO = 0.30
const KB_ = 500.0; const R0 = 0.8
const RC_LJ = 2.5; const RC_CG = 3.5
const DR = 0.01
const NBIN = floor(Int, RC_CG / DR)
const RGRID = ((0:NBIN-1) .+ 0.5) .* DR

# ------------------------------------------------------------- 원자 모형
"원자 힘과 virial W = sum r.f (분자 내 LJ는 제외, 결합은 조화 퍼텐셜)"
function fine_forces(x, L, N)
    n = 2N; F = zeros(2, n); W = 0.0; rc2 = RC_LJ^2
    @inbounds for i in 1:n, j in i+1:n
        (i + 1) ÷ 2 == (j + 1) ÷ 2 && continue                 # 같은 분자
        dx = x[1, i] - x[1, j]; dy = x[2, i] - x[2, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        r2 = dx * dx + dy * dy
        if r2 < rc2
            ir2 = 1.0 / r2; ir6 = ir2^3
            fr = (48.0 * ir6 * ir6 - 24.0 * ir6) * ir2
            F[1, i] += fr * dx; F[2, i] += fr * dy; F[1, j] -= fr * dx; F[2, j] -= fr * dy
            W += fr * r2
        end
    end
    @inbounds for m in 1:N
        i = 2m - 1; j = 2m
        dx = x[1, i] - x[1, j]; dy = x[2, i] - x[2, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        r = sqrt(dx * dx + dy * dy); fr = -KB_ * (r - R0) / r
        F[1, i] += fr * dx; F[2, i] += fr * dy; F[1, j] -= fr * dx; F[2, j] -= fr * dy
        W += fr * r * r
    end
    return F, W
end

function com(x, L, N)
    c = zeros(2, N)
    for m in 1:N
        dx = x[1, 2m] - x[1, 2m-1]; dy = x[2, 2m] - x[2, 2m-1]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        c[1, m] = x[1, 2m-1] + 0.5dx; c[2, m] = x[2, 2m-1] + 0.5dy
    end
    return c
end

function hist_add!(h, c, L, N)
    @inbounds for i in 1:N, j in i+1:N
        dx = c[1, i] - c[1, j]; dy = c[2, i] - c[2, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        b = floor(Int, sqrt(dx * dx + dy * dy) / DR) + 1
        b <= length(h) && (h[b] += 2.0)
    end
end

function run_fine(N, L, nequil, nprod, dt, gamma, seed, every)
    Random.seed!(seed)
    n = 2N; side = ceil(Int, sqrt(N)); a = L / side
    x = zeros(2, n); m = 0
    for ix in 0:side-1, iy in 0:side-1                       # 격자 위 분자, 무작위 방향
        if m < N
            m += 1; th = 2pi * rand()
            cx = (ix + 0.5) * a; cy = (iy + 0.5) * a
            x[1, 2m-1] = cx + 0.5R0 * cos(th); x[2, 2m-1] = cy + 0.5R0 * sin(th)
            x[1, 2m] = cx - 0.5R0 * cos(th); x[2, 2m] = cy - 0.5R0 * sin(th)
        end
    end
    v = sqrt(T) .* randn(2, n)
    F, W = fine_forces(x, L, N)
    c1 = exp(-gamma * dt); c2 = sqrt(1 - c1 * c1) * sqrt(T)
    h = zeros(NBIN); nfr = 0; Wsum = 0.0; Ksum = 0.0
    unw = zeros(2, N); cprev = com(x, L, N); msd_t = zeros(nprod ÷ every + 1); c0 = zeros(2, N)
    for step in 0:nequil+nprod-1
        v .+= 0.5dt .* F; x .+= 0.5dt .* v
        v .= c1 .* v .+ c2 .* randn(2, n)
        x .+= 0.5dt .* v; x .= mod.(x, L)
        F, W = fine_forces(x, L, N)
        v .+= 0.5dt .* F
        if step >= nequil
            c = com(x, L, N); d = c .- cprev; d .-= L .* round.(d ./ L)
            unw .+= d; cprev = c
            k = step - nequil
            if k % every == 0
                hist_add!(h, c, L, N); nfr += 1; Wsum += W; Ksum += 0.5 * sum(abs2, v)
                k == 0 && (c0 .= unw)
                msd_t[k÷every+1] = sum(abs2, unw .- c0) / N
            end
        else
            cprev = com(x, L, N)
        end
    end
    P = (Ksum / nfr + 0.5 * Wsum / nfr) / L^2            # 2D 원자 virial 압력 (운동 부분은 측정 K)
    return h, nfr, P, msd_t
end

# ------------------------------------------------------------- CG 모형
function cg_forces(c, L, N, Ftab)
    F = zeros(2, N); W = 0.0; nb = length(Ftab)
    @inbounds for i in 1:N, j in i+1:N
        dx = c[1, i] - c[1, j]; dy = c[2, i] - c[2, j]
        dx -= L * round(dx / L); dy -= L * round(dy / L)
        r = sqrt(dx * dx + dy * dy)
        s = r / DR - 0.5; b = trunc(Int, s)                    # 파이썬 int(): 0 쪽으로 자름
        if b < 0
            f = Ftab[1]
        elseif b >= nb - 1
            continue
        else
            w = s - b; f = (1 - w) * Ftab[b+1] + w * Ftab[b+2]
        end
        fr = f / r
        F[1, i] += fr * dx; F[2, i] += fr * dy; F[1, j] -= fr * dx; F[2, j] -= fr * dy
        W += f * r
    end
    return F, W
end

function run_cg(N, L, Ftab, nequil, nprod, dt, gamma, seed, every, mass)
    Random.seed!(seed)
    side = ceil(Int, sqrt(N)); a = L / side; c = zeros(2, N); m = 0
    for ix in 0:side-1, iy in 0:side-1
        if m < N
            m += 1; c[1, m] = (ix + 0.5) * a; c[2, m] = (iy + 0.5) * a
        end
    end
    v = sqrt(T / mass) .* randn(2, N)
    F, W = cg_forces(c, L, N, Ftab)
    c1 = exp(-gamma * dt); c2 = sqrt(1 - c1 * c1) * sqrt(T / mass)
    h = zeros(NBIN); nfr = 0; Wsum = 0.0; unw = zeros(2, N); msd_t = zeros(nprod ÷ every + 1)
    for step in 0:nequil+nprod-1
        v .+= 0.5dt .* F ./ mass
        dxh = 0.5dt .* v; c .+= dxh
        v .= c1 .* v .+ c2 .* randn(2, N)
        dx2 = 0.5dt .* v; c .+= dx2
        step >= nequil && (unw .+= dxh .+ dx2)
        c .= mod.(c, L)
        F, W = cg_forces(c, L, N, Ftab)
        v .+= 0.5dt .* F ./ mass
        if step >= nequil
            k = step - nequil
            if k % every == 0
                hist_add!(h, c, L, N); nfr += 1; Wsum += W
                msd_t[k÷every+1] = sum(abs2, unw) / N
            end
        end
    end
    P = (N * T + 0.5 * Wsum / nfr) / L^2                     # 등분배 운동 항
    return h, nfr, P, msd_t
end

# ------------------------------------------------------------- 결정론적 도우미
function gofr(h, nfr, N, L)
    rho = N / L^2
    shell = pi .* ((RGRID .+ 0.5DR) .^ 2 .- (RGRID .- 0.5DR) .^ 2)
    return h ./ (nfr * N * rho .* shell)
end

"5점 이동평균 n회 (np.convolve(np.pad(y, 2, 'edge'), ones(5)/5, 'valid'))"
function smooth(y, n = 2)
    for _ in 1:n
        yp = vcat(fill(y[1], 2), y, fill(y[end], 2))
        y = [sum(@view yp[i:i+4]) / 5 for i in 1:length(y)]
    end
    return y
end

"np.gradient(V, DR): 내부 중앙차분, 양 끝 1차 차분"
function gradient1(V, h)
    g = similar(V); n = length(V)
    g[1] = (V[2] - V[1]) / h; g[n] = (V[n] - V[n-1]) / h
    for i in 2:n-1
        g[i] = (V[i+1] - V[i-1]) / (2h)
    end
    return g
end

"핵심부 외삽, 평활, r_c에서 0으로 감쇠. (V, Ftab)"
function finalize_V(V, valid)
    V = copy(V); i0 = findfirst(valid)
    slope = min((V[i0+5] - V[i0]) / (5DR), -20.0)
    V[1:i0-1] .= V[i0] .+ slope .* (RGRID[1:i0-1] .- RGRID[i0])
    V = smooth(V, 2)
    V .-= V[end]
    taper = ones(length(V)); m = RGRID .> RC_CG - 0.5
    taper[m] .= 0.5 .* (1 .+ cos.(pi .* (RGRID[m] .- (RC_CG - 0.5)) ./ 0.5))
    V .*= taper
    return V, -gradient1(V, DR)
end

gerr(g, gt) = (m = RGRID .> 0.5; sqrt(mean((g[m] .- gt[m]) .^ 2)))

linslope(x, y) = (hcat(x, ones(length(x))) \ y)[1]

function main()
    L = sqrt(NMOL / RHO); dt = 0.005
    @printf("=== Step 1: fine MD, %d LJ dimers (bond k=%.0f, r0=%s), rho_mol=%s, L=%.2f, T=%s, dt=%s, gamma=1.0 ===\n",
            NMOL, KB_, R0, RHO, L, T, dt)
    t0 = time()
    h, nfr, P_fine, _ = run_fine(NMOL, L, 15000, 80000, dt, 1.0, 1, 40)
    g_t = gofr(h, nfr, NMOL, L)
    @printf("  frames=%d, P_fine=%.4f, wall=%.0f s\n", nfr, P_fine, time() - t0)
    ipk = argmax(g_t)
    @printf("  target g_CG(r): first peak r=%.3f, g=%.3f\n", RGRID[ipk], g_t[ipk])

    valid = g_t .> 1e-3
    V = zeros(NBIN); V[valid] .= -T .* log.(g_t[valid])      # 식 (5.4.4): PMF 초기 추정
    V, Ftab = finalize_V(V, valid)
    println("=== Step 2-3: IBI, V_{n+1} = V_n + alpha kT ln(g_n/g_target), alpha=1, CG MD 5k+30k steps ===")
    @printf("  iter   g RMSE    P_CG      (P_fine=%.4f)\n", P_fine)
    g_last = zeros(NBIN); P_ibi = 0.0
    for it in 0:8
        hc, nf, Pcg, _ = run_cg(NMOL, L, Ftab, 5000, 30000, dt, 1.0, 100 + it, 15, 2.0)
        g = gofr(hc, nf, NMOL, L); g_last = g; P_ibi = Pcg
        @printf("  %3d   %.4f   %.4f\n", it, gerr(g, g_t), Pcg)
        it == 8 && break
        m = valid .& (g .> 1e-3)
        V[m] .+= 1.0 * T .* log.(g[m] ./ g_t[m])              # 식 (5.4.5), alpha = 1
        V, Ftab = finalize_V(V, valid)
    end
    hc, nf, _, _ = run_cg(NMOL, L, Ftab, 5000, 30000, dt, 1.0, 999, 15, 2.0)   # 통계 잡음 바닥
    @printf("  noise floor (two independent CG runs, same V): g RMSE=%.4f\n", gerr(gofr(hc, nf, NMOL, L), g_last))
    F_ibi = copy(Ftab)

    println("=== Step 4: pressure-corrected IBI, dV = A (1 - r/rc), A = -0.1 kT sign(dP) min(1, 3|dP|) ===")
    Vp = copy(V); Fp = copy(Ftab)
    for it in 0:6
        hc, nf, Pcg, _ = run_cg(NMOL, L, Fp, 5000, 30000, dt, 1.0, 300 + it, 15, 2.0)
        g = gofr(hc, nf, NMOL, L)
        @printf("  %3d   g RMSE=%.4f   P_CG=%.4f\n", it, gerr(g, g_t), Pcg)
        it == 6 && break
        dP = Pcg - P_fine; Aamp = -0.1 * T * sign(dP) * min(1.0, 3.0 * abs(dP))
        m = valid .& (g .> 1e-3)
        Vp[m] .+= 1.0 * T .* log.(g[m] ./ g_t[m])
        Vp .+= Aamp .* (1 .- RGRID ./ RC_CG)
        Vp, Fp = finalize_V(Vp, valid)
    end

    println("=== Step 5: transferability, same potentials at rho_mol=0.20 ===")
    L2 = sqrt(NMOL / 0.20)
    h2, n2, P2f, _ = run_fine(NMOL, L2, 10000, 50000, dt, 1.0, 2, 40)
    g2f = gofr(h2, n2, NMOL, L2)
    for (name, Ft) in (("IBI", F_ibi), ("IBI+P", Fp))
        hc, nf, Pc, _ = run_cg(NMOL, L2, Ft, 5000, 30000, dt, 1.0, 500, 15, 2.0)
        @printf("  %-6s: g RMSE vs fine=%.4f, P_CG=%.4f, P_fine=%.4f\n", name, gerr(gofr(hc, nf, NMOL, L2), g2f), Pc, P2f)
    end

    println("=== Step 6: dynamics, weak Langevin gamma=0.05, rho_mol=0.30, 3 seeds x 30k steps ===")
    every = 50; Dfs = Float64[]; Dcs = Float64[]
    for sd in 0:2
        _, _, _, msd_f = run_fine(NMOL, L, 10000, 30000, dt, 0.05, 3 + sd, every)
        _, _, _, msd_c = run_cg(NMOL, L, F_ibi, 10000, 30000, dt, 0.05, 3 + sd, every, 2.0)
        tt = (0:length(msd_f)-1) .* every .* dt; sel = tt .> 0.25 * tt[end]
        push!(Dfs, linslope(tt[sel], msd_f[sel]) / 4); push!(Dcs, linslope(tt[sel], msd_c[sel]) / 4)
    end
    println("  per-seed D_fine: ", round.(Dfs; digits = 3), "  D_CG: ", round.(Dcs; digits = 3))
    t0 = time(); run_fine(NMOL, L, 0, 2000, dt, 1.0, 4, 40); tf = time() - t0
    t0 = time(); run_cg(NMOL, L, F_ibi, 0, 2000, dt, 1.0, 4, 15, 2.0); tc = time() - t0
    @printf("  cost per step: fine=%.3f ms, CG=%.3f ms, ratio=%.1f\n", tf / 2000 * 1e3, tc / 2000 * 1e3, tf / tc)
    @printf("  D_COM(fine)=%.3f +/- %.3f, D(CG, IBI)=%.3f +/- %.3f, speed-up D_CG/D_fine=%.2f\n",
            mean(Dfs), std(Dfs) / sqrt(3), mean(Dcs), std(Dcs) / sqrt(3), mean(Dcs) / mean(Dfs))
end

main()
