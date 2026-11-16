# lab_7_15.jl — Lab 7.15: l_max의 각도 분해능과 비용 (labs/73b_bodyorder_cost.py)
# Part A: CH4 정사면체 대 평면 사각형(Lab 7.8 Part D의 쌍, 차단 1.5 Å: C–H 간선만). 스칼라 입력 하나에서
#   중심 특징 A_l = Σ_j R(r_j) Y_l(r̂_j)의 노름. 네 C–H 거리가 같아 R(r_j)는 공통 인수이므로, 가중치와 무관한
#   |Σ_j Y_l(r̂_j)|를 보인다(l ≤ 2는 성분으로, l = 3, 4는 르장드르 덧셈 정리로). 파이썬 표의 숫자는 무작위
#   반경 가중치가 곱해진 값이라 크기는 다르고, 0인 칸과 0이 아닌 칸의 배치가 같아야 한다.
# Part B: TP 경로 수와 간선당 가중치 수는 조합 계산으로(결정론적), 시간은 l ≤ 2 손 구현으로 잰다.
#   l_max = 3은 일반 CG가 필요하므로 e3nn을 PythonCall로 불러 잰다(파일 끝 패턴).
include("so3.jl")
using Printf, Statistics

function partA()
    println("=== Part A: CH4 tetrahedral vs square planar, C-H 1.09 A, cutoff 1.5 A ===")
    r0 = 1.09
    tet = r0 / sqrt(3) .* [1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1]
    sq = r0 .* [1 -1 0 0; 0 0 1 -1; 0 0 0 0]
    for (name, X) in ("tetrahedral" => tet, "square planar" => sq)
        U = X ./ sqrt.(sum(abs2, X; dims = 1))
        comp = [norm(sum(sh(l, X[:, j]) for j in 1:4)) for l in 0:2]
        leg = [sqrt(max(legendre_power(l, U, ones(4)), 0)) for l in 0:4]
        println("  ", rpad(name, 13), " |sum_j Y_l| l=0..2 (components) ", join([@sprintf("%.4f", x) for x in comp], " "),
                " | l=0..4 (Legendre) ", join([@sprintf("%.4f", x) for x in leg], " "))
    end
    println("  l_max = 1 cannot separate them (no dipole: l=1 vanishes for both); l = 2 vanishes only for the tetrahedron, l = 3 only for the square")
end

"특징 = 16 x (l, (-1)^l), l = 0..l_max; 구면조화도 같은 irreps. |l1 - l2| ≤ l3 ≤ l1 + l2이고 패리티가 맞는 경로 수"
function tp_paths(lmax)
    irr = [(l, (-1)^l) for l in 0:lmax]
    return count(abs(l1 - l2) <= l3 <= l1 + l2 && p1 * p2 == p3 for (l1, p1) in irr, (l2, p2) in irr, (l3, p3) in irr)
end

function partB()
    println("=== Part B: cost of one equivariant convolution vs l_max (multiplicity 16) ===")
    for lmax in 0:3
        np = tp_paths(lmax)
        @printf("  l_max=%d: feature dim %4d, TP paths %3d, weights/edge %6d\n", lmax, 16 * (lmax + 1)^2, np, 16 * 16 * np)
    end
    println("  (timing: BenchmarkTools @btime on the l <= 2 layer of lab_7_14.jl, or on e3nn via PythonCall below)")
end

partA(); partB()

# ---------------------------------------------------------------- e3nn으로 l_max = 0..3 시간 재기 (PythonCall)
# using PythonCall
# torch = pyimport("torch"); o3 = pyimport("e3nn.o3"); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
# for lmax in 0:3
#     ir = o3.Irreps(pylist([(16, (l, (-1)^l)) for l in 0:lmax])); sh_ = o3.Irreps.spherical_harmonics(lmax)
#     tp = o3.FullyConnectedTensorProduct(ir, sh_, ir; shared_weights = false)
#     E = 756; x = ir.randn(E, -1); y = o3.spherical_harmonics(sh_, torch.randn(E, 3), true); w = torch.randn(E, tp.weight_numel; requires_grad = true)
#     t = minimum(@elapsed(tp(x, y, w).pow(2).sum().backward()) for _ in 1:5)
#     println(lmax, "  ", pyconvert(Int, tp.weight_numel), "  ", round(1e3t; digits = 1), " ms")
# end
