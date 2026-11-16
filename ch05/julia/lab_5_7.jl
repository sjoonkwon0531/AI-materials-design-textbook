# Lab 5.7 (Julia) — Cahn–Hilliard 미세조직의 유효 전도도와 탄성률 (native Q1 FEM)
# 파이썬 labs/55_homog.py 대응. scikit-fem 대신 **Julia로 직접 짠 주기 쌍선형 사각형(Q1) FEM**을 쓴다
# (PythonCall 다리 없음). 픽셀 하나 = Q1 요소 하나(또는 r x r개), 2x2 Gauss 적분. 계수가 요소마다
# 상수이므로 2x2 Gauss가 강성행렬·하중·평균을 모두 정확히 적분하고, 따라서 같은 이미지에서는
# scikit-fem(ElementQuad1)과 같은 이산 해를 준다(lab_5_7_xcheck.py가 같은 조립을 numpy로 재현해
# 장의 출력과 일치함을 확인).
# 주기 조건: 노드 (I, J)를 (I mod m, J mod m)로 접어 자유도를 공유한다(= P^T K P).
# 상수 모드 고정: 전도는 자유도 1개, 탄성은 강체 평행이동 2개(x, y)를 고정한다. 파이썬 판은 x 하나만
# 고정하고 SuperLU에 맡겼지만, 변형률·응력 평균은 고정 방식과 무관하다. Julia는 SPD 행렬을 CHOLMOD로 푼다.
#
# Cahn-Hilliard 초기 잡음: 기본은 Julia RNG(MersenneTwister(seed))라 미세조직이 파이썬과 다르고 표 [1]-[4]는
# 통계적으로만 비슷하다(층상 검사 [0]과 Keller-Dykhne 곱 ~10은 그대로 성립). USE_NUMPY_RNG = true로 두면
# PythonCall로 numpy의 default_rng(seed).random((N,N))을 받아 파이썬과 같은 지도를 만들고, 모든 숫자가
# 반올림 수준까지 같아야 한다.
using Printf, LinearAlgebra, SparseArrays, Statistics, Random, FFTW
const USE_NUMPY_RNG = false
USE_NUMPY_RNG && @eval using PythonCall

# ---------------------------------------------------------------- Cahn-Hilliard (Ch.2 Lab 2.14 방식)
function initial_noise(N, seed)
    if USE_NUMPY_RNG
        np = Base.invokelatest(PythonCall.pyimport, "numpy")
        a = Base.invokelatest(() -> np.random.default_rng(seed).random((N, N)))
        return Base.invokelatest(PythonCall.pyconvert, Matrix{Float64}, a)
    end
    return rand(MersenneTwister(seed), N, N)
end

function cahn_hilliard(; N = 128, mean0 = 0.0, eps = 0.03, dt = 0.02, nsteps = 800, S = 2.0, seed = 0, L = 2pi)
    k1 = collect(fftfreq(N, N / L)) .* 2pi                 # np.fft.fftfreq(N, d=L/N) * 2 pi
    K2 = k1 .^ 2 .+ (k1 .^ 2)'                              # meshgrid(indexing='ij'): 첫 첨자 = x
    denom = 1 .+ dt * S .* K2 .+ dt * eps^2 .* K2 .^ 2; gain = 1 .+ dt * S .* K2
    c = mean0 .+ 0.05 .* (2 .* initial_noise(N, seed) .- 1)
    chat = fft(c)
    for _ in 1:nsteps
        chat = (gain .* chat .- dt .* K2 .* fft(c .^ 3 .- c)) ./ denom   # 선형 강성항만 음해
        c = real.(ifft(chat))
    end
    return c
end

# ---------------------------------------------------------------- Q1 기준 요소 (2x2 Gauss)
const XA = (-1.0, 1.0, 1.0, -1.0); const YA = (-1.0, -1.0, 1.0, 1.0)   # 국소 노드 (0,0),(1,0),(1,1),(0,1)

"요소 한 변 h에 대해: 라플라스 강성 Klap(4x4), 기울기 적분 gint(4x2), 탄성 Klam/Kmu(8x8), Bint(3x8)"
function q1_reference(h)
    gp = (-1 / sqrt(3), 1 / sqrt(3)); detJ = (h / 2)^2
    Klap = zeros(4, 4); gint = zeros(4, 2); Klam = zeros(8, 8); Kmu = zeros(8, 8); Bint = zeros(3, 8)
    Dl = [1.0 1 0; 1 1 0; 0 0 0]; Dm = [2.0 0 0; 0 2 0; 0 0 1]       # D = lam*Dl + mu*Dm (Voigt, 공학 전단)
    for xi in gp, eta in gp
        dN = zeros(4, 2)
        for a in 1:4
            dN[a, 1] = XA[a] * (1 + YA[a] * eta) / 4 * (2 / h)       # dN/dx
            dN[a, 2] = YA[a] * (1 + XA[a] * xi) / 4 * (2 / h)        # dN/dy
        end
        B = zeros(3, 8)
        for a in 1:4
            B[1, 2a-1] = dN[a, 1]; B[2, 2a] = dN[a, 2]; B[3, 2a-1] = dN[a, 2]; B[3, 2a] = dN[a, 1]
        end
        Klap .+= (dN * dN') .* detJ; gint .+= dN .* detJ
        Klam .+= (B' * Dl * B) .* detJ; Kmu .+= (B' * Dm * B) .* detJ; Bint .+= B .* detJ
    end
    return Klap, gint, Klam, Kmu, Bint
end

"주기 셀: 단위 정사각형, n x n 픽셀, 픽셀마다 r x r 요소. conn[:, e] = 요소 e의 주기 자유도(노드)"
struct Cell
    n::Int; r::Int; m::Int; h::Float64
    conn::Matrix{Int}; pix::Matrix{Int}
end
function Cell(n, r = 1)
    m = n * r; conn = zeros(Int, 4, m * m); pix = zeros(Int, 2, m * m)
    node(I, J) = mod(I, m) * m + mod(J, m) + 1             # 파이썬 master = I*m + J (0-based)
    e = 0
    for ie in 0:m-1, je in 0:m-1
        e += 1
        conn[:, e] = [node(ie, je), node(ie + 1, je), node(ie + 1, je + 1), node(ie, je + 1)]
        pix[1, e] = ie ÷ r + 1; pix[2, e] = je ÷ r + 1     # 요소 중심이 속한 픽셀
    end
    return Cell(n, r, m, 1.0 / m, conn, pix)
end
elem_values(cell, img, v1, v2) = [img[cell.pix[1, e], cell.pix[2, e]] ? v2 : v1 for e in 1:size(cell.conn, 2)]

"전도도 셀 문제 (Derivation Box 5.17): div k (G + grad w) = 0, w 주기. G = x, y -> K_eff (2x2)"
function k_eff(cell, img, k1, k2)
    Klap, gint, _, _, _ = q1_reference(cell.h)
    kq = elem_values(cell, img, k1, k2); nd = cell.m^2; ne = length(kq)
    Is = Int[]; Js = Int[]; Vs = Float64[]; sizehint!(Is, 16ne); sizehint!(Js, 16ne); sizehint!(Vs, 16ne)
    for e in 1:ne, a in 1:4, b in 1:4
        push!(Is, cell.conn[a, e]); push!(Js, cell.conn[b, e]); push!(Vs, kq[e] * Klap[a, b])
    end
    K = sparse(Is, Js, Vs, nd, nd)
    F = cholesky(Symmetric(K[2:end, 2:end]))               # 자유도 1 고정 (상수 모드)
    Keff = zeros(2, 2)
    for (i, G) in enumerate(([1.0, 0.0], [0.0, 1.0]))
        rhs = zeros(nd)
        for e in 1:ne, a in 1:4
            rhs[cell.conn[a, e]] -= kq[e] * (gint[a, 1] * G[1] + gint[a, 2] * G[2])
        end
        w = zeros(nd); w[2:end] = F \ rhs[2:end]
        jx = 0.0; jy = 0.0                                 # 셀 면적 = 1이므로 적분 = 평균
        for e in 1:ne
            gx = 0.0; gy = 0.0
            for a in 1:4
                wa = w[cell.conn[a, e]]; gx += gint[a, 1] * wa; gy += gint[a, 2] * wa
            end
            jx += kq[e] * (G[1] * cell.h^2 + gx); jy += kq[e] * (G[2] * cell.h^2 + gy)
        end
        Keff[1, i] = jx; Keff[2, i] = jy
    end
    return Keff
end

lame(E, nu) = (E * nu / ((1 + nu) * (1 - 2nu)), E / (2 * (1 + nu)))

"평면 변형 탄성 셀 문제. 거시 변형률 (exx, eyy, exy) = (1,0,0), (0,1,0), (0,0,1/2) -> C_eff (3x3, Voigt)"
function C_eff(cell, img, E1, E2, nu = 0.3)
    _, _, Klam, Kmu, Bint = q1_reference(cell.h)
    l1, m1 = lame(E1, nu); l2, m2 = lame(E2, nu)
    lq = elem_values(cell, img, l1, l2); mq = elem_values(cell, img, m1, m2)
    nd = 2 * cell.m^2; ne = length(lq)
    dof(e, a) = 2 * cell.conn[(a+1)÷2, e] - isodd(a)        # 국소 자유도 a(1..8) -> 전역 (x, y 교대)
    Is = Int[]; Js = Int[]; Vs = Float64[]
    for e in 1:ne, a in 1:8, b in 1:8
        push!(Is, dof(e, a)); push!(Js, dof(e, b)); push!(Vs, lq[e] * Klam[a, b] + mq[e] * Kmu[a, b])
    end
    K = sparse(Is, Js, Vs, nd, nd)
    F = cholesky(Symmetric(K[3:end, 3:end]))               # 노드 1의 x, y 고정 (강체 평행이동)
    C = zeros(3, 3)
    for i in 1:3
        Ev = zeros(3); Ev[i] = 1.0                          # Voigt 변형률 (공학 전단 gamma = 2 exy = 1)
        rhs = zeros(nd)
        for e in 1:ne
            D = lq[e] .* [1.0 1 0; 1 1 0; 0 0 0] .+ mq[e] .* [2.0 0 0; 0 2 0; 0 0 1]
            fe = -(Bint' * (D * Ev))
            for a in 1:8
                rhs[dof(e, a)] += fe[a]
            end
        end
        w = zeros(nd); w[3:end] = F \ rhs[3:end]
        sig = zeros(3)
        for e in 1:ne
            D = lq[e] .* [1.0 1 0; 1 1 0; 0 0 0] .+ mq[e] .* [2.0 0 0; 0 2 0; 0 0 1]
            we = [w[dof(e, a)] for a in 1:8]
            sig .+= D * (Ev .* cell.h^2 .+ Bint * we)       # 요소 위 응력 적분
        end
        C[:, i] = sig
    end
    return C
end

# ---------------------------------------------------------------- 2D 경계
function bounds_cond(f2, k1, k2)
    f1 = 1 - f2
    return (f1 * k1 + f2 * k2, 1 / (f1 / k1 + f2 / k2),
            k1 + f2 / (1 / (k2 - k1) + f1 / (2 * k1)), k2 + f1 / (1 / (k1 - k2) + f2 / (2 * k2)))   # V, R, HSl, HSu
end
function bounds_bulk2d(f2, K1, G1, K2, G2)
    f1 = 1 - f2
    return (f1 * K1 + f2 * K2, 1 / (f1 / K1 + f2 / K2),
            K1 + f2 / (1 / (K2 - K1) + f1 / (K1 + G1)), K2 + f1 / (1 / (K1 - K2) + f2 / (K2 + G2)))
end
tf(b) = b ? "True" : "False"                               # 파이썬 출력과 같은 표기

function main()
    t0 = time(); n = 128; k1, k2 = 1.0, 10.0
    cell = Cell(n)
    lam_img = falses(n, n); lam_img[n÷2+1:end, :] .= true   # x에 수직인 층
    Kl = k_eff(cell, lam_img, k1, k2); V, R, _, _ = bounds_cond(0.5, k1, k2)
    println("[0] laminate check (f2=0.5, k2/k1=10)")
    @printf("    k_xx FEM = %.6f   Reuss (series)   = %.6f\n", Kl[1, 1], R)
    @printf("    k_yy FEM = %.6f   Voigt (parallel) = %.6f\n", Kl[2, 2], V)

    means = (-0.45, -0.3, -0.15, 0.0)
    imgs = Dict(m => cahn_hilliard(; N = n, mean0 = m, seed = 1) .> 0 for m in means)
    println("\n[1] CH microstructures (128^2, L=2pi, eps=0.03, dt=0.02, 800 steps, S=2, seed=1)")
    rows = []
    for m in means, swap in (false, true)
        img = swap ? .!imgs[m] : imgs[m]; f2 = mean(img)
        K = k_eff(cell, img, k1, k2); kiso = 0.5 * tr(K)
        V, R, HSl, HSu = bounds_cond(f2, k1, k2)
        push!(rows, (f2, kiso, abs(K[1, 1] - K[2, 2]) / kiso, V, R, HSl, HSu))
    end
    sort!(rows; by = first)
    println("   f2     k_eff   aniso   Reuss   HS_low  HS_up   Voigt   inside_HS")
    for (f2, kiso, an, V, R, HSl, HSu) in rows
        @printf("  %.3f  %6.3f  %5.3f  %6.3f  %6.3f  %6.3f  %6.3f   %s\n", f2, kiso, an, R, HSl, HSu, V, tf(HSl <= kiso <= HSu))
    end

    img0 = imgs[0.0]
    println("\n[2] symmetric quench (mean=0): 8 seeds, k1=1, k2=10")
    println("  seed   f2     k_xx    k_yy    k_iso   sqrt(detK(1,10)*detK(10,1))")
    kis = Float64[]
    for seed in 1:8
        img = cahn_hilliard(; N = n, mean0 = 0.0, seed = seed) .> 0
        Ka = k_eff(cell, img, k1, k2); Kb = k_eff(cell, img, k2, k1)
        push!(kis, sqrt(det(Ka)))
        @printf("  %3d  %.3f  %6.3f  %6.3f  %6.3f   %.4f\n", seed, mean(img), Ka[1, 1], Ka[2, 2], kis[end], sqrt(det(Ka) * det(Kb)))
    end
    @printf("  ensemble k_iso = %.3f +/- %.3f (sem), sample std = %.3f;  Keller-Dykhne sqrt(k1 k2) = %.4f\n",
            mean(kis), std(kis) / sqrt(length(kis)), std(kis), sqrt(k1 * k2))
    for r in (1, 2, 4)
        cr = Cell(n, r); Ka = k_eff(cr, img0, k1, k2); Kb = k_eff(cr, img0, k2, k1)
        @printf("  duality product (seed 1) with %dx%d refinement: %.4f  (exact 10)\n", r, r, sqrt(det(Ka) * det(Kb)))
    end

    crop = img0[1:4:end, 1:4:end]                           # 32 x 32 성긴 픽셀 지도 (파이썬 img0[::4, ::4])
    println("\n[3] mesh refinement at fixed geometry (32^2-pixel map, k2/k1=10)")
    println("   r  elements    k_xx        energy-type quantity decreases monotonically")
    for r in (1, 2, 4, 8)
        @printf("  %2d  %7d   %.6f\n", r, (32r)^2, k_eff(Cell(32, r), crop, k1, k2)[1, 1])
    end

    E1, E2, nu = 1.0, 10.0, 0.3
    l1, G1 = lame(E1, nu); l2, G2 = lame(E2, nu); K1, K2 = l1 + G1, l2 + G2
    @printf("\n[4] plane-strain elasticity (E2/E1=10, nu=0.3): kappa1=%.4f, kappa2=%.4f\n", K1, K2)
    println("   f2     kappa_eff  Reuss   HS_low  HS_up   Voigt   inside_HS")
    el = []; Csym = zeros(3, 3)
    for m in means, swap in (false, true)
        img = swap ? .!imgs[m] : imgs[m]; f2 = mean(img)
        C = C_eff(cell, img, E1, E2, nu); kap = 0.25 * (C[1, 1] + C[2, 2] + C[1, 2] + C[2, 1])
        V, R, HSl, HSu = bounds_bulk2d(f2, K1, G1, K2, G2)
        push!(el, (f2, kap, V, R, HSl, HSu))
        (m == 0.0 && !swap) && (Csym = C)
    end
    sort!(el; by = first)
    for (f2, kap, V, R, HSl, HSu) in el
        @printf("  %.3f  %7.4f   %6.4f  %6.4f  %6.4f  %6.4f   %s\n", f2, kap, R, HSl, HSu, V, tf(HSl <= kap <= HSu))
    end
    println("   C_eff (mean=0 map) =")
    for i in 1:3
        @printf("  [%8.4f %8.4f %8.4f]\n", Csym[i, 1], Csym[i, 2], Csym[i, 3])
    end
    @printf("\nruntime %.1f s\n", time() - t0)
end

main()
