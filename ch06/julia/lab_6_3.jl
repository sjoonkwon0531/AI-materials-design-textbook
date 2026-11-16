# Lab 6.3 (Julia) — 네 기술자를 한 실험대 위에서
# 파이썬 labs/62_desc.py 대응. ACSF와 sine 행렬은 descriptors.jl의 Julia 구현을 쓰고(ACSF는 DScribe와
# 같은 정의·순서, 기계 정밀도로 일치), SOAP와 MBTR은 성숙한 Julia 구현이 없어 PythonCall로 DScribe를 부른다.
# Part D(여러 결정·표면·클러스터의 SOAP 커널)는 구조 생성(ASE)과 SOAP(DScribe)가 모두 파이썬이라
# 핵심 계산(정규화한 power spectrum의 내적의 제곱)만 Julia에서 한다.
# 시간 측정은 Julia ACSF·sine 행렬만 의미가 있다(DScribe 호출 시간은 파이썬 판과 같다).
using LinearAlgebra, Printf, Random, Statistics
include("ch6_common.jl")
include("descriptors.jl")
include("dscribe_bridge.jl")
load_dscribe_shim(LABS)

const RC = 5.0
pyd(d) = pydict(d)
soap_cu = dscribe_desc.SOAP(; species = pylist(["Cu"]), r_cut = RC, n_max = 8, l_max = 6, sigma = 0.5, periodic = true)
mbtr_cu = dscribe_desc.MBTR(; species = pylist(["Cu"]), geometry = pyd(Dict("function" => "inverse_distance")),
    grid = pyd(Dict("min" => 0.1, "max" => 0.6, "n" => 200, "sigma" => 0.01)),
    weighting = pyd(Dict("function" => "exp", "scale" => 0.5, "threshold" => 1e-3)), normalization = "l2", periodic = true)
mbtr3_cu = dscribe_desc.MBTR(; species = pylist(["Cu"]), geometry = pyd(Dict("function" => "cosine")),
    grid = pyd(Dict("min" => -1, "max" => 1, "n" => 200, "sigma" => 0.02)),
    weighting = pyd(Dict("function" => "exp", "scale" => 0.5, "threshold" => 1e-3)), normalization = "l2", periodic = true)

sine_sorted(s) = vec(sorted_rows(sine_matrix(s)))
const DESC = [
    ("SOAP (per atom)", s -> dscribe_create(soap_cu, s), true),
    ("ACSF (per atom)", s -> acsf_all(s; species = [29]), true),
    ("MBTR k=2 (global)", s -> dscribe_create(mbtr_cu, s), false),
    ("MBTR k=3 (global)", s -> dscribe_create(mbtr3_cu, s), false),
    ("sine matrix, eigenvalues", s -> eigs(sine_matrix(s)), false),
    ("sine matrix, sorted rows", sine_sorted, false),
]
moved(s, d) = Structure(s.Z, s.R + d, s.cell)

function main()
    b0 = fcc_supercell(29, 3.61, 3)
    base = moved(b0, 0.05 * randn(MersenneTwister(11), 3, natoms(b0)))   # 108원자, 흔든 fcc Cu
    N = natoms(base); rng = MersenneTwister(5)

    # ---------- Part A
    @printf("=== Part A: rattled Cu fcc 3x3x3 (N=%d); rotate cell+atoms, translate+wrap, permute ===\n", N)
    perm = randperm(rng, N); inv = invperm(perm)
    Q = qr(randn(MersenneTwister(2), 3, 3)).Q |> Matrix; det(Q) < 0 && (Q[:, 1] .*= -1)
    tr = wrap(Structure(base.Z, Q * base.R .+ [0.77, -1.31, 2.05], Q * base.cell))
    tr = Structure(tr.Z[perm], tr.R[:, perm], tr.cell)
    @printf("  raw Cartesian coordinates (after un-permuting): max|dx| = %.3f A\n", maximum(abs.(tr.R[:, inv] - base.R)))
    for (name, f, peratom) in DESC
        d0, d1 = f(base), f(tr)
        peratom && (d1 = d1[inv, :])
        dim = peratom ? size(d0, 2) : length(d0)
        @printf("  %-26s dim = %6d   max|d| = %.1e   relative = %.1e\n", name, dim,
                maximum(abs.(d1 - d0)), maximum(abs.(d1 - d0)) / maximum(abs.(d0)))
    end

    # ---------- Part B
    println("=== Part B: smoothness. Displace along a fixed random pattern u (RMS per atom = amp) ===")
    u = randn(rng, 3, N); u ./= sqrt(mean(sum(u .^ 2; dims = 1)))
    amps = 10 .^ range(-4, -0.5; length = 15)
    ref = Dict(name => f(base) for (name, f, _) in DESC)
    for (name, f, _) in DESC
        rel = [norm(f(moved(base, a * u)) - ref[name]) / norm(ref[name]) for a in amps]
        c = hcat(ones(8), log10.(amps[1:8])) \ log10.(rel[1:8])          # 1차 맞춤의 기울기
        @printf("  %-26s rel. distance at 1e-4 A = %.2e, at 0.316 A = %.2e; log-log slope (amp<=0.01 A) = %.2f\n",
                name, rel[1], rel[end], c[2])
    end
    scan = range(0, 0.3; length = 301); h = step(scan)
    keys_ = ("SOAP (per atom)", "sine matrix, eigenvalues", "sine matrix, sorted rows")
    fs = Dict(name => f for (name, f, _) in DESC)
    prev = Dict(k => fs[k](base) for k in keys_); speed = Dict(k => Float64[] for k in keys_)
    roworder(s) = sortperm(-[norm(r) for r in eachrow(sine_matrix(s))])
    prev_order = roworder(base); nswap = 0
    for sv in scan[2:end]
        a = moved(base, sv * u)
        for k in keys_
            cur = fs[k](a); push!(speed[k], norm(cur - prev[k]) / h / norm(ref[k])); prev[k] = cur
        end
        o = roworder(a); nswap += (o != prev_order); prev_order = o
    end
    for k in keys_
        @printf("  path scan 0-0.3 A, step %.3f A: %-26s |dD/ds|/|D| median = %8.3f/A, max = %8.3f/A\n",
                h, k, median(speed[k]), maximum(speed[k]))
    end
    @printf("  sorted-row order changed in %d of %d steps\n", nswap, length(scan) - 1)

    # ---------- Part C (차원 공식과 Julia 구현의 비용)
    println("=== Part C: dimensions; cost of the Julia ACSF and sine matrix (108 atoms, best of 5) ===")
    println("  SOAP dim (1 species) = n_max(n_max+1)/2 * (l_max+1); e.g. n_max=8, l_max=6 -> ", 8 * 9 ÷ 2 * 7)
    println("  species scaling:  S   SOAP dim   ACSF dim   MBTR k2 dim   MBTR k3 dim")
    for S in 1:4
        soapdim = S * 8 * (S * 8 + 1) ÷ 2 * 7              # 다종: (S n_max)(S n_max + 1)/2 (l_max + 1)
        acsfdim = S * (1 + length(G2_PARAMS)) + S * (S + 1) ÷ 2 * length(G4_PARAMS)
        @printf("                   %d   %7d   %7d   %9d   %9d\n", S, soapdim, acsfdim, 200 * S * (S + 1) ÷ 2, 200 * S * S * (S + 1) ÷ 2)
    end
    tbest(f, s) = (f(s); minimum(@elapsed(f(s)) for _ in 1:5))          # 첫 호출(JIT) 제외
    for (name, f) in (("ACSF (Julia)", fs["ACSF (per atom)"]), ("sine matrix, eigenvalues", fs["sine matrix, eigenvalues"]))
        @printf("  time per atom, %-26s = %.3f ms\n", name, tbest(f, base) / N * 1e3)
    end
    b4 = fcc_supercell(29, 3.61, 4); big = moved(b4, 0.05 * randn(MersenneTwister(3), 3, natoms(b4)))
    for name in ("ACSF (per atom)", "sine matrix, eigenvalues")
        r = (tbest(fs[name], big) / natoms(big)) / (tbest(fs[name], base) / N)
        @printf("  per-atom cost ratio N=256 vs N=108, %-26s = %.2f\n", name, r)
    end

    # ---------- Part D (SOAP 커널: 구조는 ASE, SOAP는 DScribe, 커널은 Julia)
    println("=== Part D: SOAP kernel k = (p.p'/|p||p'|)^2 between environments (all atoms Cu, NN = 2.55 A) ===")
    ab = pyimport("ase.build"); nn = 2.55
    fcc = ab.bulk("Cu", "fcc"; a = nn * sqrt(2), cubic = true).repeat((3, 3, 3))
    envs = Any[("fcc", fcc, 0),
        ("hcp", ab.bulk("Cu", "hcp"; a = nn, c = nn * sqrt(8 / 3)).repeat((4, 4, 3)), 0),
        ("bcc", ab.bulk("Cu", "bcc"; a = nn * 2 / sqrt(3), cubic = true).repeat((4, 4, 4)), 0),
        ("sc", ab.bulk("Cu", "sc"; a = nn).repeat((4, 4, 4)), 0),
        ("diamond", ab.bulk("Cu", "diamond"; a = nn * 4 / sqrt(3), cubic = true).repeat((2, 2, 2)), 0)]
    for (nm, fn) in (("fcc(111) surf", ab.fcc111), ("fcc(100) surf", ab.fcc100))
        sl = fn("Cu"; size = (4, 4, 4), a = nn * sqrt(2), vacuum = 8.0, periodic = true)
        push!(envs, (nm, sl, argmax(pyconvert(Matrix{Float64}, sl.positions)[:, 3]) - 1))
    end
    vac = fcc.copy(); dd = pyconvert(Vector{Float64}, vac.get_distances(0, pylist(0:pyconvert(Int, pylen(vac))-1); mic = true))
    nbr = sortperm(dd)[2] - 1; vac.__delitem__(0)
    push!(envs, ("fcc vacancy nbr", vac, nbr - 1))
    for (nm, sd) in (("fcc rattled 0.10", 0.10), ("fcc rattled 0.30", 0.30))
        r_ = fcc.copy(); r_.rattle(sd; seed = 4); push!(envs, (nm, r_, 0))
    end
    ico = pyimport("ase.cluster").Icosahedron("Cu"; noshells = 3)
    P0 = pyconvert(Matrix{Float64}, ico.positions); ctr = vec(mean(P0; dims = 1))
    ci = argmin([norm(P0[i, :] - ctr) for i in 1:size(P0, 1)])
    dc = sort([norm(P0[i, :] - P0[ci, :]) for i in 1:size(P0, 1)])[2]
    ico.positions = np.array(P0[ci:ci, :] .+ (P0 .- P0[ci:ci, :]) .* (nn / dc))
    ico.center(; vacuum = 8.0); ico.pbc = true
    push!(envs, ("icosahedron ctr", ico, ci - 1))
    function kern(rc, sg; zeta = 2)
        sk = dscribe_desc.SOAP(; species = pylist(["Cu"]), r_cut = rc, n_max = 8, l_max = 6, sigma = sg, periodic = true)
        P = reduce(vcat, [permutedims(vec(pyconvert(Array{Float64}, sk.create(a; centers = pylist([i]))))) for (_, a, i) in envs])
        P ./= sqrt.(sum(P .^ 2; dims = 2))
        return (P * P') .^ zeta
    end
    Kw = kern(5.0, 0.5)
    @printf("  r_cut=5.0, sigma=0.5 (Part A-C settings): min K = %.3f, fcc-hcp = %.4f, fcc-bcc = %.4f, fcc-diamond = %.3f\n",
            minimum(Kw), Kw[1, 2], Kw[1, 3], Kw[1, 5])
    K = kern(3.5, 0.3)
    println("  r_cut=3.5, sigma=0.3 (sharper, used for the heat map):")
    for (i, (nm, _, _)) in enumerate(envs)
        @printf("    %-17s %s\n", nm, join([@sprintf("%.3f", K[i, j]) for j in 1:length(envs)], " "))
    end
    @printf("  min K = %.3f; fcc-hcp = %.4f; fcc-bcc = %.4f; fcc-(fcc rattled 0.10) = %.4f\n", minimum(K), K[1, 2], K[1, 3], K[1, 9])

    # ---------- Part E (ACSF는 Julia, SOAP는 DScribe)
    println("=== Part E: CH4 tetrahedral vs square planar (C-H = 1.09 A), descriptor of the C atom ===")
    r = 1.09
    tet = r / sqrt(3) * [1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1]
    sq = r * [1 -1 0 0; 0 0 1 -1; 0 0 0 0]
    mols = [Structure([6, 1, 1, 1, 1], hcat(zeros(3), x), nothing) for x in (tet, sq)]
    sm = dscribe_desc.SOAP(; species = pylist(["C", "H"]), r_cut = RC, n_max = 8, l_max = 6, sigma = 0.5)
    for (nm, f) in (("ACSF G2 only", m -> acsf(m, 1; species = [1, 6], g4 = Tuple{Float64,Int,Int}[])),
                    ("ACSF G2+G4", m -> acsf(m, 1; species = [1, 6])),
                    ("SOAP", m -> vec(dscribe_create(sm, m; centers = pylist([0])))))
        x, y = f(mols[1]), f(mols[2])
        @printf("  %-13s: ||d_tet - d_sq|| / ||d_tet|| = %.2e\n", nm, norm(x - y) / norm(x))
    end
end

main()
