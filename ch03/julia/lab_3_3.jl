# Lab 3.3 (Julia) — convex hull로 상안정성 판정 (Li-O 이원계)
# Python 판은 pymatgen PhaseDiagram을 쓴다. 이원계(binary)에서는 hull이
# (x_O, 원자당 형성에너지) 평면의 "하부 볼록 포락선(lower convex hull)"이므로
# Andrew의 monotone chain 알고리즘 몇 줄로 직접 세울 수 있다.
using Printf

# (표시용 이름, Li 원자수, O 원자수, 총에너지 eV) — 원소 기준 0 => 형성에너지 hull
# 표시 이름은 pymatgen의 reduced_formula 관례(O -> "O2", 과산화물 Li2O2)를 따른다.
entries = [
    ("Li",    1, 0,  0.0),
    ("O2",    0, 1,  0.0),
    ("Li2O",  2, 1, -2.05 * 3),   # ~ -2.05 eV/atom
    ("Li2O2", 2, 2, -1.58 * 4),   # ~ -1.58 eV/atom
    ("LiO2",  1, 2, -0.90 * 3),   # ~ -0.90 eV/atom
]
xO(e)    = e[3] / (e[2] + e[3])
eform(e) = e[4] / (e[2] + e[3])          # 원소 기준이 0이므로 E/atom = 형성에너지/atom

# 2D 외적: (o->a) x (o->b)
cross2(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])

"하부 볼록 포락선 (x 오름차순 점들)"
function lower_hull(pts)
    P = sort(pts)                         # x, 그다음 y 순 정렬
    H = Tuple{Float64,Float64}[]
    for p in P
        while length(H) >= 2 && cross2(H[end-1], H[end], p) <= 0
            pop!(H)                       # 오른쪽으로 꺾이거나 일직선이면 제거
        end
        push!(H, p)
    end
    return H
end

"hull 위 거리: 같은 x에서 (점의 에너지) - (hull 선분의 선형보간 에너지)"
function e_above_hull(H, x, e)
    for k in 1:length(H)-1
        (x1, e1), (x2, e2) = H[k], H[k+1]
        if x1 - 1e-12 <= x <= x2 + 1e-12
            return e - (e1 + (e2 - e1) * (x - x1) / (x2 - x1))
        end
    end
    error("x outside hull range")
end

pts = [(xO(e), eform(e)) for e in entries]
H = lower_hull(pts)
for e in entries
    x, ef = xO(e), eform(e)
    eh = e_above_hull(H, x, ef)
    tag = eh < 1e-8 ? "STABLE" : @sprintf("%+.3f eV/atom above hull", eh)
    @printf("  %-6s x_O=%.3f E_form=%+.3f eV/atom -> %s\n", e[1], x, ef, tag)
end
println("hull vertices (x_O, E_form): ", [(round(p[1]; digits=3), round(p[2]; digits=3)) for p in H])

# (선택) pymatgen을 PythonCall.jl로 그대로 호출할 수도 있다:
# using PythonCall
# pdm = pyimport("pymatgen.analysis.phase_diagram"); C = pyimport("pymatgen.core").Composition
# ents = pylist([pdm.PDEntry(C(f), E) for (f, E) in [("Li",0.0),("O",0.0),("Li2O",-6.15),("Li2O2",-6.32),("LiO2",-2.7)]])
# pd = pdm.PhaseDiagram(ents); println(pd.get_e_above_hull(ents[4]))   # Python 0-based 인덱스: LiO2
