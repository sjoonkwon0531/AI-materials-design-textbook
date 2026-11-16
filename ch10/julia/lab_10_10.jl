# lab_10_10.jl: Lab 10.10: MACE-MP-0으로 이완하기, 대칭 제약, 빈자리, 그리고 선이완의 손익 (labs/106_relax.py)
# 실행: julia lab_10_10.jl [A][B][C]   (기본 ABC; MACE와 CHGNet이 같은 인터프리터 v10에 있다)
# 에너지와 힘은 PythonCall로 부르는 ASE 계산기(MACE-MP-0 small float64, CHGNet 0.3.0)에서 온다.
# 원자 위치만 움직이는 이완(Part B, Part C의 빈자리)은 ASE 3.29와 같은 식의 BFGS를 Julia로 옮긴 bfgs()로 한다.
# 셀까지 움직이는 이완(FrechetCellFilter)과 대칭 제약(FixSymmetry)은 ASE에 맡긴다(ase_relax).
# 흔든 출발 구조: FROM_PYTHON_START = true이면 파이썬 판이 저장한 data/106_sto_rattled_start.xyz, 106_vac64_start.xyz를 읽어
# 같은 출발점에서 시작한다. false이면 Julia 난수로 흔든다(ASE rattle의 numpy 난수열은 재현되지 않는다).
ENV["CH10_MODEL"] = "mace"
include("ch10_common2.jl")
const PARTS = isempty(ARGS) ? "ABC" : ARGS[1]
const FROM_PYTHON_START = true
read_xyz(f) = cell_from_ase(pyimport("ase.io").read(joinpath(DATA, f)))

const mace = Counted(make_calc("mace"; dtype = "float64")[1])

# ------------------------------------------------------------------ Part A: 입방정 SrTiO₃, a = 4.10 Å에서 출발
function partA()
    println("\n=== Part A: cubic SrTiO3, start a = 4.10 A (exp 3.905, PBE 3.944) ===")
    start = sto(4.10)
    rattled = FROM_PYTHON_START ? read_xyz("106_sto_rattled_start.xyz") : Cell(start.Z, start.R .+ 0.03 .* randn(Xoshiro(0), 3, 5), start.h)
    res = Dict{String,Any}()
    for (tag, s0, sym) in (("sym-constrained", start, true), ("free, rattled", rattled, false))
        t0 = time(); r = ase_relax(mace, s0; fmax = 1e-3, sym)
        L = cellpar(r.cell.h)
        @printf("  %-16s: steps %3d, %5.2f s, E = %.5f eV, a,b,c = %.4f %.4f %.4f A, angles %.2f %.2f %.2f, space group %s\n",
                tag, r.steps, time() - t0, r.E, L[1], L[2], L[3], L[4], L[5], L[6], r.sg)
        res[tag] = r
    end
    @printf("  E(free) - E(cubic) = %.2f meV per 5-atom cell\n", 1000 * (res["free, rattled"].E - res["sym-constrained"].E))
    fr = res["free, rattled"].cell
    dTiO = sort([norm(mic(fr.R[:, o] .- fr.R[:, 2], fr.h)) for o in 3:5])      # Ti(2번)와 세 O
    println("  free: Ti-O distances = ", join([@sprintf("%.4f", x) for x in dTiO], " "), " A")
    acub = cellpar(res["sym-constrained"].cell.h)[1]
    @printf("  cubic a = %.4f A : %+.2f%% vs exp 3.905, %+.2f%% vs PBE(VASP) 3.944\n", acub, 100 * (acub / 3.905 - 1), 100 * (acub / 3.944 - 1))
    println("  force tolerance (sym-constrained, from a = 4.10 A):")
    for fm in (0.05, 0.01, 0.001, 0.0001)
        r = ase_relax(mace, start; fmax = fm, sym = true)
        p = -tr(r.stress) / 3 * GPA                                      # 압력 = -tr σ / 3
        @printf("    fmax %7.4f eV/A: steps %3d, a = %.5f A, residual pressure %+.4f GPa\n", fm, r.steps, cellpar(r.cell.h)[1], p)
    end
    return res
end

# ------------------------------------------------------------------ Part B: Si 중성 단일 빈자리
"빈자리 형성 에너지 E_f = E(N-1) - (N-1) ε_bulk, 이웃 넷 사이의 거리 여섯 개, 최대 변위"
function vacancy(N_rep, e_si, si0; start = nothing, efun)
    sc = repeat_cell(si0, N_rep); N = natoms(sc)
    vac = delete_atom(sc, 1)                                    # 원점의 원자(1번)를 지운다
    Eu = efun(vac)[1]
    d0 = [norm(mic(sc.R[:, j] .- sc.R[:, 1], sc.h)) for j in 2:N]
    nbr = sortperm(d0)[1:4]                                     # 지운 자리의 이웃 넷 (vac에서의 번호)
    pert = if start === nothing
        R = copy(vac.R); R[:, nbr] .+= 0.02 .* randn(Xoshiro(1), 3, 4); Cell(vac.Z, R, vac.h)   # Td를 깨 D2d가 나타날 여지를 준다
    else
        start
    end
    t0 = time(); rel = bfgs(efun, pert; fmax = 0.005)
    Ef_u = Eu - (N - 1) * e_si; Ef_r = rel.E - (N - 1) * e_si
    pairs = sort([norm(mic(rel.cell.R[:, nbr[p]] .- rel.cell.R[:, nbr[q]], rel.cell.h)) for p in 1:4 for q in p+1:4])
    disp = maximum(norm.(eachcol(rel.cell.R .- vac.R)))
    @printf("  N = %3d: Ef(unrelaxed) = %.3f eV, Ef(relaxed) = %.3f eV, relaxation %+.3f eV, steps %d, %.1f s\n",
            N, Ef_u, Ef_r, Ef_r - Ef_u, rel.steps, time() - t0)
    @printf("           neighbour-neighbour distances = %s A (ideal %.3f); max displacement %.3f A\n",
            join([@sprintf("%.3f", x) for x in pairs], " "), sqrt(2) * si0.h[1, 1] / 2, disp)
    return rel
end
function partB()
    println("\n=== Part B: neutral Si monovacancy (MACE-MP-0 small) ===")
    r = ase_relax(mace, si_cubic(5.43); fmax = 1e-4, sym = true)
    si0 = r.cell; e_si = r.E / 8
    @printf("  bulk Si: a = %.4f A, E/atom = %.5f eV\n", si0.h[1, 1], e_si)
    efun(s) = energy_forces(mace, s)
    rel64 = vacancy(2, e_si, si0; start = FROM_PYTHON_START ? read_xyz("106_vac64_start.xyz") : nothing, efun)
    vacancy(3, e_si, si0; efun)
    return rel64
end

# ------------------------------------------------------------------ Part C: MACE로 선이완, 기준 모델(CHGNet, DFT의 대역)로 마무리
function partC()
    chg = Counted(make_calc("chgnet")[1])
    println("\n=== Part C: pre-relax with MACE, finish with the reference (CHGNet 0.3.0 as the stand-in for DFT) ===")
    # 출발점: 흔든 SrTiO₃(셀도 움직인다)와 흔든 63원자 빈자리(원자만). 선이완 구조는 파이썬 판이 저장한 MACE 이완 결과다.
    # (lab_10_10.jl의 Part A, B 결과를 쓰려면 아래 두 파일 대신 그 구조를 넘기면 된다.)
    cases = (("SrTiO3 free, rattled", "106_sto_rattled_start.xyz", "106_sto_free.xyz", true),
             ("Si vacancy 63 atoms", "106_vac64_start.xyz", "106_vac64_mace.xyz", false))
    for (tag, f_raw, f_pre, cellopt) in cases
        raw, pre = read_xyz(f_raw), read_xyz(f_pre)
        if cellopt
            r1 = ase_relax(chg, raw; fmax = 0.01); r2 = ase_relax(chg, pre; fmax = 0.01); rm = ase_relax(mace, raw; fmax = 0.01)
        else
            ef_c = s -> energy_forces(chg, s); ef_m = s -> energy_forces(mace, s)
            r1 = bfgs(ef_c, raw; fmax = 0.01); r2 = bfgs(ef_c, pre; fmax = 0.01); rm = bfgs(ef_m, raw; fmax = 0.01)
        end
        @printf("  %-22s: reference calls from raw start %3d | MACE calls %3d then reference calls %3d | final E difference %+.2f meV\n",
                tag, r1.steps + 1, rm.steps + 1, r2.steps + 1, 1000 * (r2.E - r1.E))
    end
end

t0 = time()
occursin('A', PARTS) && partA()
occursin('B', PARTS) && partB()
occursin('C', PARTS) && partC()
@printf("\ntotal %.1f s, MACE calls %d\n", time() - t0, mace.n)
