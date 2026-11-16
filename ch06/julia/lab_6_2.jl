# Lab 6.2 (Julia) — Coulomb 행렬의 불변성과 두 가지 실패
# 파이썬 labs/62_cm.py 대응. Coulomb 행렬과 sine 행렬은 descriptors.jl에 직접 구현했다.
# 분자 좌표는 ASE g2 집합의 값을 상수로 옮겼다(에탄올, 메탄). 단위: Z는 기본 전하, 거리는 Å.
# 결정론적 부분(Part A 행 노름·고유값 불변성, Part C 교차점 s*와 표, Part E 완전 결정의 sine 고유값)은
# 숫자가 그대로 일치해야 한다. 회전·순열·잡음·최적화 시작점은 Julia 난수를 쓰므로 Part B, D와
# Part E의 흔든 NaCl 숫자는 통계적으로만 일치한다.
using LinearAlgebra, Printf, Random, Statistics
include("descriptors.jl")

const Z_ETOH = [6, 6, 8, 1, 1, 1, 1, 1, 1]
const R_ETOH = [ 1.168181 -0.400382  0.0
                 0.0       0.559462  0.0
                -1.190083 -0.227669  0.0
                -1.946623  0.381525  0.0
                 0.042557  1.207508  0.886933
                 0.042557  1.207508 -0.886933
                 2.115891  0.1448    0.0
                 1.128599 -1.037234  0.885881
                 1.128599 -1.037234 -0.885881]' |> Matrix          # 3 x 9
const R_CH4 = [0.0 0.0 0.0; 0.629118 0.629118 0.629118; -0.629118 -0.629118 0.629118;
               0.629118 -0.629118 -0.629118; -0.629118 0.629118 -0.629118]' |> Matrix

"균일 무작위 회전 행렬 (Gaussian 행렬의 QR, det = +1로 맞춤)"
function random_rotation(rng)
    F = qr(randn(rng, 3, 3)); Q = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    det(Q) < 0 && (Q[:, 1] .*= -1)
    return Q
end

"Levenberg-Marquardt (중심 차분 야코비안). scipy least_squares 대신 쓰는 짧은 구현."
function levmar(f, x0; iters = 400, h = 1e-7)
    x = copy(x0); r = f(x); lam = 1e-3
    for _ in 1:iters
        J = reduce(hcat, [(f(x + h * e) - f(x - h * e)) / (2h) for e in eachcol(Matrix(1.0I, length(x), length(x)))])
        A = J' * J; g = J' * r
        dx = -((A + lam * Diagonal(diag(A) .+ 1e-12)) \ g)
        rn = f(x + dx)
        if sum(abs2, rn) < sum(abs2, r)
            x += dx; r = rn; lam = max(lam / 3, 1e-15)
        else
            lam *= 4
        end
        (maximum(abs, r) < 1e-13 || lam > 1e12) && break
    end
    return x, r
end

pairdist(R, idx) = sort([norm(R[:, i] - R[:, j]) for i in idx for j in idx if i < j])

function main()
    rng = MersenneTwister(7)
    Z, R = Z_ETOH, R_ETOH; N = length(Z)
    # ---------- Part A
    Q = random_rotation(rng); perm = randperm(rng, N)
    R2 = (Q * R .+ [1.7, -3.2, 0.4])[:, perm]; Z2 = Z[perm]
    M1, M2 = coulomb_matrix(Z, R), coulomb_matrix(Z2, R2)
    @printf("=== Part A: ethanol (N=%d), random rotation + translation + permutation ===\n", N)
    @printf("  raw Cartesian coordinates : max|dx|      = %.3f A\n", maximum(abs.(R2 - R)))
    @printf("  raw CM (no sorting)       : max|dM|      = %.3f\n", maximum(abs.(M2 - M1)))
    @printf("  sorted-row CM             : max|dM|      = %.2e\n", maximum(abs.(sorted_rows(M2) - sorted_rows(M1))))
    @printf("  eigenvalue spectrum       : max|dlambda| = %.2e\n", maximum(abs.(eigs(M2) - eigs(M1))))
    println("  row norms of M (atoms CCOHHHHHH): " * join([@sprintf("%.3f", norm(r)) for r in eachrow(M1)], " "))

    # ---------- Part B
    println("=== Part B: randomized CM (sigma = 1), 5 augmented copies of the same ethanol ===")
    nrm = [norm(r) for r in eachrow(M1)]; base = vec(sorted_rows(M1))
    dists = Float64[]; orders = Set{Vector{Int}}()
    for _ in 1:5
        p = sortperm(-(nrm .+ randn(rng, N))); push!(orders, p)
        push!(dists, norm(vec(M1[p, p]) - base))
    end
    @printf("  distinct row orders = %d / 5;  ||M_rand - M_sorted||_F = %s   (||M_sorted||_F = %.1f)\n",
            length(orders), join([@sprintf("%.1f", x) for x in dists], ", "), norm(base))

    # ---------- Part C
    iO = findfirst(==(8), Z)
    iH = argmin([Z[j] == 1 ? norm(R[:, j] - R[:, iO]) : 9.0 for j in 1:N])
    u = (R[:, iH] - R[:, iO]) / norm(R[:, iH] - R[:, iO])
    Hidx = [j for j in 1:N if Z[j] == 1 && j != iH]
    geom(s) = (X = copy(R); X[:, iH] += s * u; X)
    norms(s) = [norm(r) for r in eachrow(coulomb_matrix(Z, geom(s)))]
    ss = range(0.0, 1.5; length = 3001)
    gap = [norms(s)[iH] - norms(s)[j] for s in ss, j in Hidx]
    k0, jj = 0, 0
    for k in 1:length(ss)-1, j in 1:length(Hidx)            # 파이썬 np.where(...)[0][0]: 행 우선 첫 번째
        if sign(gap[k+1, j]) != sign(gap[k, j])
            k0, jj = k, j; break
        end
    end
    a, b = ss[k0], ss[k0+1]
    f(s) = norms(s)[iH] - norms(s)[Hidx[jj]]
    for _ in 1:80
        c = 0.5 * (a + b)
        sign(f(c)) == sign(f(a)) ? (a = c) : (b = c)
    end
    s_star = 0.5 * (a + b)
    @printf("=== Part C: stretch hydroxyl O-H by s (A); its row norm crosses a C-H hydrogen at s* = %.4f A ===\n", s_star)
    println("     eps      ||sorted(s*+eps)-sorted(s*-eps)||   ||eig(s*+eps)-eig(s*-eps)||")
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
        Mp, Mm = coulomb_matrix(Z, geom(s_star + eps)), coulomb_matrix(Z, geom(s_star - eps))
        @printf("  %8.0e   %12.4f                    %.2e\n", eps, norm(sorted_rows(Mp) - sorted_rows(Mm)), norm(eigs(Mp) - eigs(Mm)))
    end

    # ---------- Part D
    println("=== Part D: a 5-atom C1H4 geometry and a different one with the SAME eigenspectrum ===")
    Zm = [6, 1, 1, 1, 1]
    R0 = R_CH4 + 0.12 * randn(rng, 3, 5)
    lam0 = eigs(coulomb_matrix(Zm, R0))
    resid(x) = eigs(coulomb_matrix(Zm, reshape(x, 3, 5))) - lam0
    best = nothing
    for trial in 1:20
        x, r = levmar(resid, vec(R_CH4 + 0.35 * randn(rng, 3, 5)))
        R1 = reshape(x, 3, 5)
        d0 = sort([norm(R0[:, j] - R0[:, 1]) for j in 2:5]); d1 = sort([norm(R1[:, j] - R1[:, 1]) for j in 2:5])
        hh0, hh1 = pairdist(R0, 2:5), pairdist(R1, 2:5)
        diff = max(maximum(abs.(d1 - d0)), maximum(abs.(hh1 - hh0)))
        if maximum(abs, r) < 1e-10 && (best === nothing || diff > best[1])
            best = (diff, R1, d0, d1, hh0, hh1, maximum(abs, r))
        end
    end
    if best === nothing
        println("  no exact twin found in 20 starts; try more starts or another seed")
    else
        diff, R1, d0, d1, hh0, hh1, res = best
        fmt = v -> join([@sprintf("%.3f", x) for x in v], " ")
        println("  degrees of freedom 3N-6 = 9 > N = 5 eigenvalues")
        @printf("  max|lambda(R1) - lambda(R0)| = %.1e\n", res)
        println("  sorted C-H distances R0: ", fmt(d0)); println("  sorted C-H distances R1: ", fmt(d1))
        println("  sorted H-H distances R0: ", fmt(hh0)); println("  sorted H-H distances R1: ", fmt(hh1))
        @printf("  largest difference in a sorted distance = %.3f A  -> different molecules, same descriptor\n", diff)
        @printf("  sorted-row CM distance between them = %.3f\n", norm(sorted_rows(coulomb_matrix(Zm, R1)) - sorted_rows(coulomb_matrix(Zm, R0))))
    end

    # ---------- Part E
    println("=== Part E: rattled NaCl conventional cell (8 atoms, sigma = 0.1 A), shift origin by t and wrap ===")
    a0 = 5.64; h = a0 / 2
    Zc = [11, 17, 11, 17, 11, 17, 11, 17]                     # ASE bulk("NaCl", "rocksalt", cubic=True) 순서
    Rc = [0 0 0; h 0 0; 0 h h; h h h; h 0 h; 0 0 h; h h 0; 0 h 0]' |> Matrix{Float64}
    B = Matrix(a0 * I(3))
    nacl = Structure(Zc, Rc + 0.1 * randn(MersenneTwister(1), 3, 8), B)
    ref_c, ref_s = eigs(coulomb_matrix(Zc, nacl.R)), eigs(sine_matrix(nacl))
    for t in ([0.3, 0.0, 0.0], [1.1, 2.0, -0.7], [2.82, 2.82, 2.82])
        sh = wrap(Structure(Zc, nacl.R .+ t, B))
        dc = maximum(abs.(eigs(coulomb_matrix(Zc, sh.R)) - ref_c))
        ds = maximum(abs.(eigs(sine_matrix(sh)) - ref_s))
        @printf("  t = (%5.2f,%5.2f,%5.2f) A:  cell-CM eig max|d| = %8.3f   sine-matrix eig max|d| = %.1e\n", t[1], t[2], t[3], dc, ds)
    end
    ev_conv = eigs(sine_matrix(Structure(Zc, Rc, B)))
    Bp = [0 h h; h 0 h; h h 0]' |> Matrix{Float64}            # 원시 세포(열 = 격자 벡터)
    ev_prim = eigs(sine_matrix(Structure([11, 17], [0 h; 0 0; 0 0], Bp)))
    println("  same crystal, primitive cell (2 atoms): sine eig = ", round.(ev_prim; digits = 2))
    println("                conventional cell (8 atoms): sine eig = ", round.(ev_conv; digits = 2))

    # ---------- Part F
    BLAS.set_num_threads(1)
    println("=== Part F: cost of the sine-matrix eigenspectrum vs cell size (Cu fcc supercells, best of 5, 1 BLAS thread) ===")
    eigs(sine_matrix(fcc_supercell(29, 3.61, 2)))             # JIT 예열
    for rep in 2:6
        at = fcc_supercell(29, 3.61, rep)
        at = Structure(at.Z, at.R + 0.05 * randn(MersenneTwister(1), size(at.R)...), at.cell)
        ts = [(@elapsed eigs(sine_matrix(at))) for _ in 1:5]
        @printf("  N = %4d: total %8.2f ms, per atom %.4f ms\n", natoms(at), minimum(ts) * 1e3, minimum(ts) / natoms(at) * 1e3)
    end
end

main()
