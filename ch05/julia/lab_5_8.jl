# Lab 5.8 (Julia) — D2Q9 BGK: Poiseuille 검증과 원기둥 주위 흐름
# 파이썬 labs/55_lbm.py의 Numba 커널(step_numba)과 같은 알고리즘: Guo 힘, 당기기(pull) 흐름,
# 중간 반사(halfway bounce-back) 벽. 격자 단위(dx = dt = 1). 결정론적이므로 모든 숫자가 그대로 일치해야 한다
# (lab_5_8_xcheck.py가 이 파일의 1-based 인덱스 규약을 numpy로 옮겨 ny=8, 16 결과를 대조).
# 배열 배치: f[i, x, y] (i = 속도 방향 1..9가 가장 빠르게 변하는 첨자 -> column-major에서 연속).
# 파이썬 [0] 항목(numpy 판 대 numba 판 비교)은 같은 커널 두 벌을 맞춰 보는 검사라 옮기지 않았다.
using Printf

const CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const W9 = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
const OPP = (1, 4, 5, 2, 3, 8, 9, 6, 7)        # 파이썬 opp = [0,3,4,1,2,7,8,5,6]에 1을 더한 것
const CS2 = 1 / 3

function feq!(f, rho, ux, uy)
    nx, ny = size(rho)
    for y in 1:ny, x in 1:nx, i in 1:9
        cu = CX[i] * ux[x, y] + CY[i] * uy[x, y]; usq = ux[x, y]^2 + uy[x, y]^2
        f[i, x, y] = W9[i] * rho[x, y] * (1 + 3cu + 4.5cu * cu - 1.5usq)
    end
    return f
end

"충돌(BGK + Guo 힘) 후 당기기 흐름. 고체 노드는 건너뛰고, 고체에서 올 성분은 중간 반사로 채운다."
function step!(fn, fpost, f, tau, Fx, Fy, solid)
    nx, ny = size(solid)
    @inbounds for y in 1:ny, x in 1:nx
        solid[x, y] && continue
        rho = 0.0; mx = 0.0; my = 0.0
        for i in 1:9
            rho += f[i, x, y]; mx += f[i, x, y] * CX[i]; my += f[i, x, y] * CY[i]
        end
        ux = (mx + 0.5Fx[x, y]) / rho; uy = (my + 0.5Fy[x, y]) / rho     # Guo: 반 힘 이동
        usq = ux * ux + uy * uy
        for i in 1:9
            cu = CX[i] * ux + CY[i] * uy
            fe = W9[i] * rho * (1 + 3cu + 4.5cu * cu - 1.5usq)           # f_eq, c_s^2 = 1/3
            Si = W9[i] * ((3 * (CX[i] - ux) + 9cu * CX[i]) * Fx[x, y] + (3 * (CY[i] - uy) + 9cu * CY[i]) * Fy[x, y])
            fpost[i, x, y] = f[i, x, y] - (f[i, x, y] - fe) / tau + (1 - 0.5 / tau) * Si
        end
    end
    fill!(fn, 0.0)
    @inbounds for y in 1:ny, x in 1:nx
        solid[x, y] && continue
        for i in 1:9
            xs = mod1(x - CX[i], nx); ys = mod1(y - CY[i], ny)
            fn[i, x, y] = solid[xs, ys] ? fpost[OPP[i], x, y] : fpost[i, xs, ys]   # 중간 반사
        end
    end
    return fn
end

"거시량: 고체 노드는 rho = 1로 두고(분포가 0), ux = sum f c / rho + F/(2 rho)"
function macros(f, Fx, Fy)
    rho = dropdims(sum(f; dims = 1); dims = 1); rho[rho .<= 0] .= 1.0
    ux = dropdims(sum(f .* reshape(collect(CX), 9, 1, 1); dims = 1); dims = 1) ./ rho .+ 0.5 .* Fx ./ rho
    uy = dropdims(sum(f .* reshape(collect(CY), 9, 1, 1); dims = 1); dims = 1) ./ rho .+ 0.5 .* Fy ./ rho
    return rho, ux, uy
end

"채널: 유체 행 y = 2..ny+1, 고체 행 y = 1, ny+2. 벽은 그 중간(y = 0.5, ny + 0.5, 0-based 좌표)"
function poiseuille(ny, tau, g; tol = 1e-12, maxit = 400000, nx = 3)
    Ny = ny + 2
    solid = falses(nx, Ny); solid[:, 1] .= true; solid[:, end] .= true
    Fx = [solid[x, y] ? 0.0 : g for x in 1:nx, y in 1:Ny]; Fy = zeros(nx, Ny)
    f = feq!(zeros(9, nx, Ny), [solid[x, y] ? 0.0 : 1.0 for x in 1:nx, y in 1:Ny], zeros(nx, Ny), zeros(nx, Ny))
    fn = similar(f); fpost = zeros(size(f))
    m0 = sum(f); uold = Float64[]; drift = 0.0; u = Float64[]; it = 0
    for k in 1:maxit
        it = k
        step!(fn, fpost, f, tau, Fx, Fy, solid); f, fn = fn, f
        if k % 200 == 0
            drift = max(drift, abs(sum(f) - m0) / m0)
            _, ux, _ = macros(f, Fx, Fy)
            u = ux[2, 2:end-1]                                  # 파이썬 ux[1, 1:-1]
            (!isempty(uold) && maximum(abs.(u .- uold)) < tol * maximum(abs.(u))) && break
            uold = u
        end
    end
    nu = CS2 * (tau - 0.5)
    yy = collect(1:ny) .- 0.5                                   # 벽(y = 0.5)으로부터의 거리
    ua = g / (2nu) .* yy .* (ny .- yy)
    err = sqrt(sum((u .- ua) .^ 2) / sum(ua .^ 2))
    return yy, u, ua, err, it, drift
end

function main()
    t0 = time()
    println("[1] Poiseuille, ny=16, g chosen so that u_max = 0.01 (Ma ~ 0.017)")
    println("   tau     nu       L2 rel err    iters   max mass drift   slip*nu/g   (16tau^2-16tau+1)/24")
    for tau in (0.55, 0.6, 0.7, 0.8, 0.9, 0.933, 1.0, 1.2, 1.5, 2.0)
        nu = CS2 * (tau - 0.5); ny = 16; g = 0.01 * 8nu / ny^2
        yy, u, ua, err, it, dr = poiseuille(ny, tau, g)
        slip = sum(u .- ua) / length(u) * nu / g               # u - ua는 균일한 이동(미끄럼)
        @printf("  %5.3f  %.4f   %.3e    %6d   %.1e       %+.5f    %+.5f\n",
                tau, nu, err, it, dr, slip, (16tau^2 - 16tau + 1) / 24)
    end
    println("\n[2] grid convergence at tau=0.8 (u_max=0.01 fixed, Ma fixed)")
    println("   ny     L2 rel err    order")
    prev = NaN
    for ny in (8, 16, 32, 64)
        tau = 0.8; nu = CS2 * (tau - 0.5); g = 0.01 * 8nu / ny^2
        err = poiseuille(ny, tau, g)[4]
        order = isnan(prev) ? "" : @sprintf("%.2f", log2(prev / err))
        @printf("  %3d    %.3e    %s\n", ny, err, order); prev = err
    end

    println("\n[3] cylinder in a periodic channel (body-force driven), two forcing levels")
    nx, ny, R, tau = 200, 60, 8, 0.6
    Ny = ny + 2; nu = CS2 * (tau - 0.5)
    # 파이썬 0-based 좌표 X, Y로 마스크를 정의하고 1-based 배열에 저장
    solid = [(Y == 0) || (Y == Ny - 1) || ((X - nx ÷ 4)^2 + (Y - Ny / 2 + 0.5)^2 < R^2) for X in 0:nx-1, Y in 0:Ny-1]
    @printf("   grid %dx%d, D=%d, tau=%s, nu=%.4f, 25000 steps each\n", nx, ny, 2R, tau, nu)
    for g in (2.0e-6, 2.0e-5)
        Fx = [solid[x, y] ? 0.0 : g for x in 1:nx, y in 1:Ny]; Fy = zeros(nx, Ny)
        f = feq!(zeros(9, nx, Ny), [solid[x, y] ? 0.0 : 1.0 for x in 1:nx, y in 1:Ny], zeros(nx, Ny), zeros(nx, Ny))
        fn = similar(f); fpost = zeros(size(f)); m0 = sum(f)
        for _ in 1:25000
            step!(fn, fpost, f, tau, Fx, Fy, solid); f, fn = fn, f
        end
        _, ux, uy = macros(f, Fx, Fy); ux[solid] .= 0; uy[solid] .= 0
        U = mean_(ux[.!solid]); Re = U * 2R / nu
        umax = maximum(sqrt.(ux .^ 2 .+ uy .^ 2))
        yc = round(Int, Ny / 2 - 0.5)                           # 0-based 중심 행
        xs = (nx ÷ 4 + R):(nx - 1)                              # 0-based 원기둥 뒤쪽 열
        rev = [X for X in xs if ux[X+1, yc+1] < 0]
        L_rec = isempty(rev) ? 0 : maximum(rev) - (nx ÷ 4 + R) + 1
        @printf("   g=%.1e: mean u=%.3e, max|u|=%.3e (Ma=%.3f), Re_D=%.1f, mass drift=%.1e, recirculation length=%d lu (%.2f D)\n",
                g, U, umax, umax / sqrt(CS2), Re, abs(sum(f) - m0) / m0, L_rec, L_rec / (2R))
    end
    @printf("\nruntime %.1f s\n", time() - t0)
end

mean_(v) = sum(v) / length(v)

main()
