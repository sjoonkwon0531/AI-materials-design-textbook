# Lab 4.7 (Julia) — H/Pt(111) 흡착: 자리 비교 + 층수·진공 수렴 시험 (ASE + EMT, PythonCall.jl)
# 파이썬 판(lab_4_7.py)과 같은 함수 구조. 이완·에너지 계산은 모두 ASE(파이썬) 안에서
# 일어나므로, 같은 ASE 버전이면 숫자가 그대로 재현된다.
using PythonCall, Printf

build    = pyimport("ase.build")
EMT      = pyimport("ase.calculators.emt").EMT
BFGS     = pyimport("ase.optimize").BFGS
FixAtoms = pyimport("ase.constraints").FixAtoms
pyimport("warnings").filterwarnings("ignore")

fl(x) = pyconvert(Float64, x)
zcoords(atoms) = pyconvert(Matrix{Float64}, atoms.get_positions())[:, 3]

function relax(atoms; fmax = 0.05)
    atoms.calc = EMT(); BFGS(atoms, logfile = nothing).run(fmax = fmax, steps = 300)
    return fl(atoms.get_potential_energy())
end

h2 = build.molecule("H2"); h2.calc = EMT(); BFGS(h2, logfile = nothing).run(fmax = 0.01)
const eH2 = fl(h2.get_potential_energy())              # 이완한 H2 기준

function E_ads_site(site; size = (2, 2, 4), vacuum = 8.0, h = 1.4)
    slab = build.fcc111("Pt", size = size, vacuum = vacuum)
    layers = sort(unique(round.(zcoords(slab); digits = 2)))
    cut = layers[2] + 0.1                               # 바닥 2층 고정선 (1-based [2] = 파이썬 [1])
    base = slab.copy(); base.set_constraint(FixAtoms(mask = pylist(zcoords(base) .< cut)))
    E_clean = relax(base)
    sl = slab.copy(); build.add_adsorbate(sl, "H", height = h, position = site)
    sl.set_constraint(FixAtoms(mask = pylist(zcoords(sl) .< cut)))
    return relax(sl) - E_clean - 0.5 * eH2             # E_ads = E(slab+H) - E(slab) - E(H2)/2
end

@printf("H2 reference (EMT) = %.4f eV\n", eH2)
sites = ["fcc", "hcp", "bridge", "ontop"]
Eads = Dict(s => E_ads_site(s) for s in sites)
for s in sites
    @printf("  %-8s %+.4f\n", s, Eads[s])
end
best = argmin(s -> Eads[s], sites)                      # Julia 1.7+: 함수값이 최소인 원소
@printf("  -> most stable site (EMT): %s  (%.4f eV)\n", best, Eads[best])
for n in (3, 4, 5, 6)
    @printf("  layers %d: %+.4f\n", n, E_ads_site("fcc"; size = (2, 2, n)))
end
for v in (5, 7, 9, 12)
    @printf("  vacuum %d: %+.4f\n", v, E_ads_site("fcc"; vacuum = Float64(v)))
end
