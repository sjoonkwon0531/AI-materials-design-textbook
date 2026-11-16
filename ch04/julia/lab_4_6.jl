# Lab 4.6A (Julia) — ASE + EMT 표준 워크플로를 PythonCall.jl로 구동
# ASE는 파이썬 생태계의 표준 도구이고 EMT calculator도 ASE 안에 구현되어 있어, Julia로
# "네이티브 이식"하면 같은 결과를 내는 다른 코드를 새로 짜는 셈이 된다. 대신 PythonCall.jl로
# ASE 객체를 그대로 다룬다 — 이것 자체가 §4.6의 교훈(구조와 calculator의 분리, 인터페이스의 재사용)이다.
# 난수(MaxwellBoltzmann, Langevin, rattle)는 모두 파이썬 쪽 numpy RandomState로 생성하므로,
# 파이썬 판과 같은 ASE 버전이면 출력 숫자가 그대로 재현된다.
#   준비:  ] add PythonCall ;  using CondaPkg; CondaPkg.add_pip("ase")
using PythonCall, Printf, Statistics

np        = pyimport("numpy")
build     = pyimport("ase.build")
EMT       = pyimport("ase.calculators.emt").EMT
BFGS      = pyimport("ase.optimize").BFGS
EOSfit    = pyimport("ase.eos").EquationOfState
units     = pyimport("ase.units")
mbd       = pyimport("ase.md.velocitydistribution").MaxwellBoltzmannDistribution
Langevin  = pyimport("ase.md.langevin").Langevin
FixAtoms  = pyimport("ase.constraints").FixAtoms
FCCcluster = pyimport("ase.cluster.cubic").FaceCenteredCubic
warnings  = pyimport("warnings"); warnings.filterwarnings("ignore")
np.random.seed(0)

jl(M) = pyconvert(Matrix{Float64}, M)                # numpy 2D 배열 -> Julia Matrix (shape 유지)
fl(x) = pyconvert(Float64, x)
fmax_of(atoms) = sqrt(maximum(sum(jl(atoms.get_forces()) .^ 2; dims = 2)))   # 원자별 |F|의 최댓값

println("--- (1) Calculator abstraction ---")
n2 = build.molecule("N2"); n2.calc = EMT()           # <- 힘의 출처를 붙이는 한 줄
@printf("  N2 potential energy = %.4f eV\n", fl(n2.get_potential_energy()))
println("  N2 forces (eV/Ang):")
show(stdout, "text/plain", round.(jl(n2.get_forces()); digits = 4) .+ 0.0); println()

println("\n--- (2) EOS: Cu fcc ---")
cu = build.bulk("Cu", "fcc", a = 3.6, cubic = true); cell0 = cu.get_cell()
vols = Float64[]; ens = Float64[]
for s in range(0.94, 1.06; length = 11)
    c = cu.copy(); c.set_cell(pymul(cell0, s), scale_atoms = true); c.calc = EMT()   # pymul: 파이썬 쪽 곱셈 (Cell * s)
    push!(vols, fl(c.get_volume())); push!(ens, fl(c.get_potential_energy()))
end
eos = EOSfit(pylist(vols), pylist(ens))
v0, e0, B = pyconvert(Tuple{Float64,Float64,Float64}, eos.fit())
a_eq = (v0 / pylen(cu) * 4)^(1 / 3)
B_GPa = B / fl(units.kJ) * 1.0e24
@printf("  V0=%.3f Ang^3, a=%.4f Ang, B=%.1f GPa\n", v0, a_eq, B_GPa)

println("\n--- (3) BFGS: Cu cluster ---")
clus = FCCcluster("Cu", surfaces = pylist([(1, 0, 0), (1, 1, 1)]), layers = pylist([1, 1]), latticeconstant = 3.6)
clus.rattle(0.1, seed = 1); clus.calc = EMT()
e_init = fl(clus.get_potential_energy()); fmax_init = fmax_of(clus)
BFGS(clus, logfile = nothing).run(fmax = 0.02, steps = 200)      # nothing -> None
e_final = fl(clus.get_potential_energy()); fmax_final = fmax_of(clus)
@printf("  atoms: %d | E_init=%.4f (fmax=%.3f) -> E_final=%.4f (fmax=%.4f); dE=%.4f\n",
        pylen(clus), e_init, fmax_init, e_final, fmax_final, e_init - e_final)

println("\n--- (4) Langevin NVT MD, Cu 4x4x4 ---")
sc = build.bulk("Cu", "fcc", a = a_eq).repeat((4, 4, 4)); sc.calc = EMT()
mbd(sc, temperature_K = 300, rng = np.random.RandomState(0))
dyn = Langevin(sc, timestep = 2.0 * fl(units.fs), temperature_K = 300, friction = 0.02,
               fixcm = false, rng = np.random.RandomState(1))
Ts = Float64[]; Es = Float64[]
function rec()                                       # Julia 함수를 파이썬 콜백으로 등록
    push!(Ts, fl(sc.get_temperature()))
    push!(Es, fl(sc.get_potential_energy()) + fl(sc.get_kinetic_energy()))
    return nothing
end
dyn.attach(pyfunc(rec), interval = 10); dyn.run(1500)
h = length(Ts) ÷ 2
@printf("  atoms=%d mean T(2nd half)=%.1f +/- %.1f K; E drift=%+.3f eV\n",
        pylen(sc), mean(Ts[h+1:end]), std(Ts[h+1:end]; corrected = false), Es[end] - Es[1])

println("\n--- (5) Pt(111) slab + H ---")
slab = build.fcc111("Pt", size = (2, 2, 4), vacuum = 7.0)
slabH = slab.copy(); build.add_adsorbate(slabH, "H", height = 1.5, position = "fcc"); slabH.calc = EMT()
zs = jl(slab.get_positions())[:, 3]                  # z 좌표 = 3번째 열 (1-based)
cutz = sort(unique(round.(zs; digits = 3)))[2]       # 아래에서 두 번째 층 (파이썬의 [1])
fixmask(atoms) = pylist(jl(atoms.get_positions())[:, 3] .< cutz + 0.1)   # 바닥 2층 고정
slab.set_constraint(FixAtoms(mask = fixmask(slab))); slab.calc = EMT()
BFGS(slab, logfile = nothing).run(fmax = 0.05, steps = 200); e_slab = fl(slab.get_potential_energy())
slabH.set_constraint(FixAtoms(mask = fixmask(slabH)))
BFGS(slabH, logfile = nothing).run(fmax = 0.05, steps = 200); e_slabH = fl(slabH.get_potential_energy())
h2 = build.molecule("H2"); h2.calc = EMT(); BFGS(h2, logfile = nothing).run(fmax = 0.01)
eH2 = fl(h2.get_potential_energy())
@printf("  slab=%.4f slab+H=%.4f H2=%.4f  E_ads=%.4f eV\n", e_slab, e_slabH, eH2, e_slabH - e_slab - 0.5 * eH2)
println("  pbc=[", join(pyconvert(Vector{Bool}, slab.pbc), ", "), "], cell=",   # join: Bool[1,1,0] 대신 true, true, false
        round.(pyconvert(Vector{Float64}, slab.cell.lengths()); digits = 3))
