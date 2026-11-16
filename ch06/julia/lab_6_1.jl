# Lab 6.1 (Julia) — 조성 기술자는 다형체를 보지 못한다
# 파이썬 labs/61_polymorph.py 대응.
#  Part A: Cu의 네 구조(fcc, hcp, bcc, sc)의 EMT 에너지. EMT 계산 자체는 ASE를 PythonCall로 부르고,
#          격자상수 1차원 최적화는 Optim.jl의 Brent 방법(scipy minimize_scalar 'bounded'와 같은 계열)으로 한다.
#  Part B: expt_gap 사본이 조성 하나당 한 행임을 Julia 화학식 파서로 확인한다.
#  Part C: Magpie 특성화 비용 (matminer를 PythonCall로 부름).
using Printf, Random, Statistics, CSV, DataFrames, Optim
include("ch6_common.jl")
include("descriptors.jl")
include("dscribe_bridge.jl")

const asebuild = pyimport("ase.build")
function cu_energy(kind, a)
    at = kind == "hcp" ? asebuild.bulk("Cu", "hcp"; a = a, c = a * sqrt(8 / 3)) : asebuild.bulk("Cu", kind; a = a)
    at.calc = pyimport("ase.calculators.emt").EMT()
    return pyconvert(Float64, at.get_potential_energy()) / pyconvert(Int, pylen(at))
end

function main()
    println("=== Part A: Cu polymorphs, EMT energy at optimized lattice constant ===")
    ranges = [("fcc", 3.3, 3.9), ("hcp", 2.3, 2.8), ("bcc", 2.6, 3.2), ("sc", 2.2, 2.8)]
    E = Dict{String,Float64}()
    for (s, lo, hi) in ranges
        r = optimize(a -> cu_energy(s, a), lo, hi, Brent(); abs_tol = 1e-5)
        E[s] = Optim.minimum(r)
        @printf("  %-4s a = %.4f A   E_EMT = %8.1f meV/atom\n", s, Optim.minimizer(r), 1000 * E[s])
    end
    println("  relative to fcc (meV/atom): " *
            join([@sprintf("%s %+.1f", s, 1000 * (E[s] - E["fcc"])) for (s, _, _) in ranges], ", "))
    spread = 1000 * (maximum(values(E)) - minimum(values(E)))
    @printf("  any composition-only model must err by >= %.1f meV/atom on at least one of them\n", spread / 2)

    println("=== Part B: expt_gap (matbench, CrabNet mirror) ===")
    d = CSV.read(joinpath(LABS, "expt_gap_matbench.csv"), DataFrame)
    red = reduced_key.(d.formula)
    nel = countmap(length.(first.(parse_formula.(d.formula))))
    @printf("  rows = %d, unique reduced formulas = %d, metals (gap = 0) = %d (%.1f%%)\n",
            nrow(d), length(unique(red)), sum(d.target .== 0), 100 * mean(d.target .== 0))
    println("  number of elements per formula: " * join(["$k: $(nel[k])" for k in sort(collect(keys(nel)))], ", "))

    println("=== Part C: Magpie featurization cost (300 random formulas, 1 process) ===")
    ep = pyimport("matminer.featurizers.composition").ElementProperty.from_preset("magpie")
    Comp = pyimport("pymatgen.core").Composition
    idx = randperm(MersenneTwister(0), nrow(d))[1:300]
    comps = [Comp(d.formula[i]) for i in idx]
    t0 = time()
    X = reduce(vcat, [permutedims(pyconvert(Vector{Float64}, ep.featurize(c))) for c in comps])
    dt = time() - t0
    @printf("  %.2f ms per formula; feature matrix %s; NaN entries = %d\n", dt / 300 * 1e3, size(X), count(isnan, X))
end

main()
