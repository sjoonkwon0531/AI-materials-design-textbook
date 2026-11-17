# lab_10_11.jl: Lab 10.11: MACE-MP-0 small을 열어 보기, 대칭, 수용장, 체수 (labs/106_dissect.py)
# 실행: julia lab_10_11.jl
# 모델 자체는 PythonCall로 연 torch 객체다. (A) 아키텍처는 그 객체의 속성을 읽고 매개변수를 블록별로 센다.
# (B) 대칭 시험의 변환(회전, 병진, 순열)과 오차 지표, (C) 거리별 묶음, (D) 다체 전개(포함–배제)와 대조군 LJ는 Julia로 옮겼다.
# (D)의 층별 원자 에너지는 두 판독 블록에 forward hook(Julia 함수를 파이썬 호출 객체로 넘긴다)을 걸어 꺼낸다.
# 난수: (B)의 변형, 흔들기, 회전과 (C)의 흔들기는 Julia 난수라 파이썬과 다른 구조다. (D)의 클러스터는 파이썬 판의
# 좌표(np.random.default_rng(11))를 그대로 적어 두었으므로 같은 구조다.
ENV["CH10_MODEL"] = "mace"
include("ch10_common2.jl")

const MP = joinpath(DATA, "mace_mp0_small.model")
const calc64 = first(make_calc("mace"; dtype = "float64"))
const net = calc64.models[0]
pyitem(x) = pyhasattr(x, "item") ? x.item() : x                 # torch 텐서, 0차원 ndarray → 파이썬 수
pyint(x) = pyconvert(Int, pyitem(x)); pyf(x) = pyconvert(Float64, pyitem(x)); pys(x) = pyconvert(String, pystr(x))
nparam(mod) = sum(pyint(p.numel()) for p in mod.parameters(); init = 0)

# ------------------------------------------------------------------ Part A: 아키텍처
function partA()
    println("=== Part A: architecture of MACE-MP-0 small ===")
    it0, pr0 = net.interactions[0], net.products[0]
    sc0 = pr0.symmetric_contractions.contractions[0]
    rmax, T = pyf(net.r_max), pyint(net.num_interactions)
    @printf("  elements: %d   cutoff r_max = %.1f A   interaction layers T = %d   receptive field T*r_max = %.1f A\n",
            pyint(pybuiltins.len(net.atomic_numbers)), rmax, T, T * rmax)
    @printf("  radial basis: %d Bessel functions, polynomial cutoff p = %.1f\n",
            pyint(net.radial_embedding.bessel_fn.bessel_weights.numel()), pyf(net.radial_embedding.cutoff_fn.p))
    println("  edge spherical harmonics: ", pys(it0.edge_attrs_irreps), "   (l_max = ", pyint(it0.edge_attrs_irreps.lmax), ")")
    println("  node features passed between layers: ", pys(pr0.linear.irreps_out))
    println("  messages inside a layer (A-basis): ", pys(it0.conv_tp.irreps_out.simplify()))
    ν = pyint(sc0.correlation)
    println("  correlation order nu = $ν  ->  body order of one layer = nu + 1 = $(ν + 1)")
    @printf("  avg_num_neighbors (message normalisation) = %.2f\n", pyf(it0.avg_num_neighbors))
    tot = nparam(net)
    @printf("  parameters total %d\n", tot)
    for kv in net.named_children()
        name, mod = pys(kv[0]), kv[1]; n = nparam(mod)
        n > 0 && @printf("    %-18s %10d  (%4.1f%%)\n", name, n, 100n / tot)
    end
    its = collect(net.interactions); prs = collect(net.products)
    skip = sum(nparam(it.skip_tp) for it in its); radial = sum(nparam(it.conv_tp_weights) for it in its)
    lin = sum(nparam(it.linear) + nparam(it.linear_up) for it in its); symc = sum(nparam(pr.symmetric_contractions) for pr in prs)
    @printf("    of which: element-dependent skip (self-connection) TP %d (%.1f%%), symmetric contractions %d (%.1f%%), radial MLPs %d (%.1f%%), equivariant linears %d (%.1f%%)\n",
            skip, 100skip / tot, symc, 100symc / tot, radial, 100radial / tot, lin, 100lin / tot)
end

# ------------------------------------------------------------------ Part B: 대칭 시험 (40원자 SrTiO₃, 흔들고 변형)
function efs_cell(calc, s::Cell)
    E, F, S = calc_efs(calc, s.Z, s.R, s.h; stress = true)
    return E, F, S
end
function partB()
    println("\n=== Part B: symmetry tests, rattled + strained SrTiO3 2x2x2 (40 atoms) ===")
    rng = Xoshiro(0)
    c0 = sto(3.95; rep = 2)
    Fm = I3 .+ 0.02 .* randn(rng, 3, 3)                         # 무작위 변형(대칭이 아닌 부분도 포함), 원자도 함께 옮긴다
    cub = Cell(c0.Z, Fm * c0.R .+ 0.05 .* randn(rng, 3, 40), Fm * c0.h)
    Rot = random_rotation(rng)
    for dt in ("float64", "float32")
        calc = dt == "float64" ? calc64 : first(make_calc("mace"; dtype = "float32"))
        E, F, S = efs_cell(calc, cub)
        Er, Fr, Sr = efs_cell(calc, Cell(cub.Z, Rot * cub.R, Rot * cub.h))           # 셀과 원자를 함께 돌린다
        tr = cub.R .+ [0.37, -1.21, 2.05]; tr = cub.h * mod.(cub.h \ tr, 1.0)        # 병진 뒤 셀 안으로 접는다
        Et, Ft, _ = efs_cell(calc, Cell(cub.Z, tr, cub.h))
        perm = randperm(rng, 40)
        Ep, Fp, _ = efs_cell(calc, Cell(cub.Z[perm], cub.R[:, perm], cub.h))
        fs = maximum(abs, F); ss = maximum(abs, S)
        @printf("  %s: |dE| rotation %.1e eV, translation %.1e eV, permutation %.1e eV (E = %.4f eV)\n", dt, abs(Er - E), abs(Et - E), abs(Ep - E), E)
        @printf("           force  rel. err: rotation (F -> R F) %.1e, translation %.1e, permutation %.1e;  |sum_i F_i| = %.1e eV/A\n",
                maximum(abs, Fr .- Rot * F) / fs, maximum(abs, Ft .- F) / fs, maximum(abs, Fp .- F[:, perm]) / fs, maximum(abs, sum(F; dims = 2)))
        @printf("           stress rel. err: rotation (S -> R S R^T) %.1e   (max|F| %.3f eV/A, max|S| %.3f GPa)\n",
                maximum(abs, Sr .- Rot * S * Rot') / ss, fs, ss * GPA)
    end
end

# ------------------------------------------------------------------ Part C: 수용장 (Si 2×2×6, 192원자)
"MACE 계산기가 결과에 남기는 원자 에너지 ε_i(node_energy)와 힘"
function node_energies(s::Cell)
    a = to_ase(s); a.calc = calc64
    F = permutedims(jarr(Float64, a.get_forces()))
    return F, jarr(Float64, calc64.results["node_energy"])
end
function partC()
    println("\n=== Part C: receptive field, Si 2x2x6 conventional cells (192 atoms, 10.9 x 10.9 x 32.8 A) ===")
    s0 = si_cubic(5.4646; rep = (2, 2, 6))
    si = Cell(s0.Z, s0.R .+ 0.01 .* randn(Xoshiro(3), size(s0.R)), s0.h)       # 힘이 대칭 때문에 0이 되지 않도록
    t0 = time()
    F0, e0 = node_energies(si)
    R1 = copy(si.R); R1[1, 1] += 0.01
    F1, e1 = node_energies(Cell(si.Z, R1, si.h))
    el = time() - t0
    dF = norm.(eachcol(F1 .- F0)); de = abs.(e1 .- e0)
    r = distances_from(si, 1)
    for (lo, hi) in ((0.0, 0.1), (0.1, 6.0), (6.0, 12.0), (12.0, 19.0))
        m = (r .>= lo) .& (r .< hi)
        @printf("  %4.1f <= r < %4.1f A : %4d atoms, max |d eps_i| = %.2e eV, max |dF_i| = %.2e eV/A\n", lo, hi, count(m), maximum(de[m]), maximum(dF[m]))
    end
    @printf("  farthest atom with |d eps_i| > 1e-12 eV: r = %.3f A; with |dF_i| > 1e-10 eV/A: r = %.3f A\n",
            maximum(r[de .> 1e-12]), maximum(r[dF .> 1e-10]))
    @printf("  wall time for two 192-atom calls %.1f s\n", el)
end

# ------------------------------------------------------------------ Part D: 6원자 Si 클러스터의 다체 전개
"파이썬 판의 좌표: 정팔면체(꼭짓점 ±1.6 Å) + N(0, 0.15²) (np.random.default_rng(11)). 열 = 원자"
const CLUSTER = [0.005128915087977625 -0.07654606151815012 1.685458953635794 -1.8770987198461164 0.10205676799112191 0.06946652378963801;
                 0.20396213104649424 -0.04469542666596706 -0.008409665856842639 0.23498231620492807 1.579515049903476 -1.4763229708704833;
                 1.78370816178789 -1.679107628955014 0.11203284243848159 -0.014464824023343082 -0.05686478506122799 -0.030379480604017725]

"ASE LennardJones(σ = 2.1, ε = 0.1, rc = 10, smooth = false): 쌍마다 4ε[(σ/r)¹² - (σ/r)⁶] - e₀, e₀는 rc에서의 값"
function lj_energy(R; sigma = 2.1, epsilon = 0.1, rc = 10.0)
    e0 = 4epsilon * ((sigma / rc)^12 - (sigma / rc)^6); E = 0.0
    for i in 1:size(R, 2), j in i+1:size(R, 2)
        r = norm(R[:, i] .- R[:, j])
        r < rc && (E += 4epsilon * ((sigma / r)^12 - (sigma / r)^6) - e0)
    end
    return E
end

"""포함–배제(Derivation Box 10.9): ε(S) = Σ_{∅≠T⊆S} (-1)^{|S|-|T|} E(T). 부분집합은 비트 마스크로 센다.
E[mask]는 마스크가 고른 원자만 남긴 계의 에너지. 반환: E_n = Σ_{|S|=n} ε(S), n = 1..N"""
function mbe(E::Dict{Int,Float64}, N)
    En = zeros(N)
    for S in 1:(1 << N) - 1
        eps = 0.0; T = S
        while T > 0                                          # S의 공집합이 아닌 부분 마스크를 모두 돈다
            eps += (-1)^(count_ones(S) - count_ones(T)) * E[T]
            T = (T - 1) & S
        end
        En[count_ones(S)] += eps
    end
    return En
end

function partD()
    println("\n=== Part D: many-body expansion of a 6-atom Si cluster ===")
    grab = Dict{Int,Py}()
    hooks = [net.readouts[i].register_forward_hook((m, inp, o) -> (grab[i] = Py(o).detach().clone(); nothing)) for i in 0:1]   # Py(): PythonCall이 텐서 인자를 PyIterable로 감싸 넘긴다
    scale = pyf(net.scale_shift.scale)
    function layer_energies(R)
        a = to_ase(fill(14, size(R, 2)), R, 30.0 .* I3; pbc = false); a.calc = calc64; a.get_potential_energy()
        return [scale * sum(jarr(Float64, grab[i].numpy())[:, 1]) for i in 0:1]
    end
    d = [norm(CLUSTER[:, i] .- CLUSTER[:, j]) for i in 1:6 for j in i+1:6]
    @printf("  pair distances %.2f-%.2f A (all inside r_max)\n", minimum(d), maximum(d))
    E1, E2, Elj = Dict{Int,Float64}(), Dict{Int,Float64}(), Dict{Int,Float64}()
    for S in 1:63
        idx = [k for k in 1:6 if (S >> (k - 1)) & 1 == 1]
        l1, l2 = layer_energies(CLUSTER[:, idx])
        E1[S] = l1; E2[S] = l2; Elj[S] = lj_energy(CLUSTER[:, idx])
    end
    for h in hooks
        h.remove()
    end
    println("  n-body increments E_n (eV), summed over all n-atom subsets:")
    println("     n :", join([lpad(n, 12) for n in 1:6]))
    for (name, E) in (("layer1", E1), ("layer2", E2), ("LJ", Elj))
        En = mbe(E, 6)
        println("  ", rpad(name, 6), ":", join([lpad(@sprintf("%.2e", x), 12) for x in En]), @sprintf("   total %.4f", sum(En)))
    end
end

t0 = time()
partA(); partB(); partC(); partD()
@printf("\ntotal %.1f s\n", time() - t0)
