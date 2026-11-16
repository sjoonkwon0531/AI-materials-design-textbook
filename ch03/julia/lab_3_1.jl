# Lab 3.1 (Julia) — 구조·조성 다루기: Si 다이아몬드 구조와 Fe2O3 조성
# Python 판은 pymatgen(Structure.from_spacegroup, SpacegroupAnalyzer, Composition)을 쓴다.
# 여기서는 같은 양을 Base + LinearAlgebra만으로 "처음부터" 계산한다.
#   - Fd-3m(No.227) 관용 단위격자 = FCC 병진 4개 x 기저 2개(0,0,0),(1/4,1/4,1/4) -> 8 sites
#   - 밀도 = (원자량 합 x amu) / 부피
#   - 공간군 확인: 48개 입방 점연산 W(부호 있는 순열행렬)와 1/4 격자 병진 t를 전수 조사해
#     구조를 자기 자신으로 보내는 대칭연산 (W,t)의 개수를 센다 -> 192 = |Fd-3m| (관용 셀 기준)
using LinearAlgebra, Printf

const AMU_G = 1.66053906660e-24          # 1 amu in g
const MASS = Dict("Si" => 28.0855, "Fe" => 55.845, "O" => 15.9994)   # pymatgen과 같은 원자량 (amu)

# ---------- (1) Si 다이아몬드 구조 ----------
a = 5.43                                   # Si 격자상수 (Angstrom)
lat = a * Matrix{Float64}(I, 3, 3)         # 입방 격자: 열(column)이 격자벡터 a1,a2,a3
fcc_t = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
basis = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
frac = [mod.(t .+ b, 1.0) for t in fcc_t for b in basis]   # 분율좌표 8개

wrap(v) = mod.(round.(v; digits=8), 1.0)   # [0,1)로 되접기 (부동소수점 잡음 제거)
key(v) = Tuple(round.(wrap(v); digits=6) .+ 0.0)   # .+ 0.0: -0.0 -> 0.0 (Set은 isequal 기준이라 -0.0 != 0.0)
siteset = Set(key(f) for f in frac)

vol = abs(det(lat))                                       # A^3
mass_g = length(frac) * MASS["Si"] * AMU_G
density = mass_g / (vol * 1e-24)                          # g/cm^3

# 대칭연산 전수 조사: W = 부호 있는 3x3 순열행렬(48개), t = (i/4, j/4, k/4)
function signed_perms()
    Ws = Matrix{Int}[]
    for p in ((1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)), s in Iterators.product((-1,1), (-1,1), (-1,1))
        W = zeros(Int, 3, 3)
        for r in 1:3
            W[r, p[r]] = s[r]
        end
        push!(Ws, W)
    end
    return Ws
end
nops = 0
for W in signed_perms(), i in 0:3, j in 0:3, k in 0:3
    t = [i, j, k] ./ 4
    if all(key(W * f .+ t) in siteset for f in frac)
        global nops += 1
    end
end

println("reduced formula : Si")
@printf("n_sites         : %d\n", length(frac))
@printf("density (g/cm^3): %.3f\n", density)
@printf("symmetry ops    : %d  (= order of Fd-3m No.227 in the conventional cell, 48 x 4)\n", nops)
@printf("lattice a (A)   : %.2f | volume (A^3): %.1f\n", a, vol)

# ---------- (2) Fe2O3 조성 ----------
comp = Dict("Fe" => 2, "O" => 3)
weight = sum(MASS[el] * n for (el, n) in comp)
xFe = comp["Fe"] / sum(values(comp))
println()
println("reduced formula : Fe2O3")
@printf("molar weight    : %.2f g/mol\n", weight)
@printf("atomic frac Fe  : %.3f\n", xFe)

# ---------- (선택) pymatgen을 PythonCall.jl로 직접 호출해 공간군 기호까지 확인 ----------
# using PythonCall
# mg  = pyimport("pymatgen.core"); sym = pyimport("pymatgen.symmetry.analyzer")
# si  = mg.Structure.from_spacegroup("Fd-3m", mg.Lattice.cubic(5.43), pylist(["Si"]), pylist([pylist([0, 0, 0])]))
# sga = sym.SpacegroupAnalyzer(si)
# println(sga.get_space_group_symbol(), " ", sga.get_space_group_number())   # Fd-3m 227
