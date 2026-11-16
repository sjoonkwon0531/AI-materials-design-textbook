# lab_7_8.jl — Lab 7.8: 구조를 그래프로 — 주기 이미지, 순열, 거리만으로는 안 보이는 것 (labs/72b_graph.py)
# Part A: pymatgen get_neighbor_list 대신 주기 이미지를 직접 도는 이웃 목록(Si 다이아몬드, 관습·원시 세포).
# Part B: 에탄올 결합 그래프의 라플라시안 스펙트럼 (GraphNeuralNetworks.jl의 GNNGraph로도 같은 그래프를 만든다).
# Part C: 학습 전 SchNet의 순열 등변성과 합 판독의 불변성. Part D: CH4 정사면체 대 평면 사각형.
include("ch7_common.jl")
include("mlip_common.jl")
import GraphNeuralNetworks as GNN

"주기 이웃 목록: (i, j, 이미지 n, 거리) — 셀 B의 열이 격자 벡터, X는 3 x N 직교 좌표. 0 < d < rc인 모든 (j, n)"
function neighbor_list(X, B, rc; nmax = 3)
    out = Tuple{Int,Int,NTuple{3,Int},Float64}[]
    for i in axes(X, 2), j in axes(X, 2), a in -nmax:nmax, b in -nmax:nmax, c in -nmax:nmax
        d = norm(X[:, j] + B * [a, b, c] - X[:, i])
        0 < d < rc && push!(out, (i, j, (a, b, c), d))
    end
    return out
end

function partA()
    println("=== Part A: Si diamond graphs (a = 5.43 A)")
    a = 5.43
    fcc = [0 0 0; 0 0.5 0.5; 0.5 0 0.5; 0.5 0.5 0]'
    conv = (a .* hcat(fcc, fcc .+ 0.25), Matrix(a * I(3)))                # 8원자 관습 세포
    prim = (a .* [0 0.25; 0 0.25; 0 0.25], a / 2 .* [0 1 1; 1 0 1; 1 1 0]) # 2원자 원시 세포(열 = 격자 벡터)
    for (name, (X, B)) in ("conventional" => conv, "primitive" => prim), rc in (2.5, 4.0)
        nl = neighbor_list(X, B, rc)
        pairs = Dict{Tuple{Int,Int},Int}()
        for (i, j, _, _) in nl
            pairs[(i, j)] = get(pairs, (i, j), 0) + 1
        end
        nself = count(e -> e[1] == e[2], nl)
        dists = sort(unique(round.(last.(nl); digits = 3)))
        @printf("%-12s N=%d r_c=%.1f A: directed edges %4d, per atom %2d, distinct (i,j) %3d, max parallel %d, self-image %d, distances %s\n",
                name, size(X, 2), rc, length(nl), length(nl) ÷ size(X, 2), length(pairs), maximum(values(pairs)), nself, dists)
    end
end

function partB()
    println("=== Part B: ethanol graphs (frame 0 of the 300 K xTB trajectory)")
    R, Z, _, _ = load_ethanol(300)
    X = R[:, :, 1]
    D = [norm(X[:, i] - X[:, j]) for i in 1:9, j in 1:9]
    for rc in (1.7, 2.5, 5.0)
        @printf("r_c=%.1f A: directed edges %3d (complete graph = 72)\n", rc, count(d -> 0 < d < rc, D))
    end
    A = Float64.((D .< 1.7) .& (D .> 0))
    L = Diagonal(vec(sum(A; dims = 2))) - A
    ev = eigvals(Symmetric(L))
    println("bond-graph Laplacian eigenvalues: ", round.(ev; digits = 3), " | # zero modes = ", count(<(1e-10), ev))
    A16 = Float64.((D .< 1.6) .& (D .> 0)); ev16 = eigvals(Symmetric(Diagonal(vec(sum(A16; dims = 2))) - A16))
    @printf("1.6 A threshold: C-C distance %.3f A -> # zero modes = %d (graph splits)\n", D[1, 2], count(<(1e-10), ev16))
    # 같은 결합 그래프를 GraphNeuralNetworks.jl로: 간선 목록(src -> dst)에서 GNNGraph를 만든다
    s, t = Int[], Int[]
    for i in 1:9, j in 1:9
        A[i, j] > 0 && (push!(s, j); push!(t, i))
    end
    g = GNN.GNNGraph(s, t; num_nodes = 9)
    Ag = Matrix(GNN.adjacency_matrix(g))
    @printf("GNNGraph: %d nodes, %d directed edges, adjacency matches: %s\n", g.num_nodes, g.num_edges, Ag == A)
    return R, Z
end

function partC(R, Z)
    println("=== Part C: permutation equivariance (untrained SchNet; Float64 input)")
    model = DenseSchNet(Z .+ 1; nf = 32, nrbf = 20, rc = 5.0, ninter = 3)
    ps, st = Lux.setup(Xoshiro(0), model); ps = to64(ps)
    X = R[:, :, 1:1]
    perm = [2, 1, 3, 6, 4, 5, 9, 7, 8]                                     # 두 탄소 교환, H 순환
    mP = DenseSchNet(Z[perm] .+ 1; nf = 32, nrbf = 20, rc = 5.0, ninter = 3)
    h = atom_features(model, X, ps, st)[:, :, 1]
    hp = atom_features(mP, X[:, perm, :], ps, st)[:, :, 1]
    E = first(model(X, ps, st))[1]; Ep = first(mP(X[:, perm, :], ps, st))[1]
    @printf("max|h(Px) - P h(x)| = %.1e   |E(Px) - E(x)| = %.1e\n", maximum(abs.(hp - h[:, perm])), abs(Ep - E))
    rd = (sum = x -> vec(sum(x; dims = 2)), mean = x -> vec(mean(x; dims = 2)), max = x -> vec(maximum(x; dims = 2)))
    println("readouts invariant? ", join(["$k $(isapprox(f(h), f(hp)))" for (k, f) in pairs(rd)], ", "),
            "; concat (order-dependent) ", isapprox(vec(h), vec(hp)))
end

function partD()
    println("=== Part D: CH4, tetrahedral vs square planar, r_CH = 1.09 A")
    r = 1.09
    tet = r / sqrt(3) .* [1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1]
    sq = r .* [1 0 -1 0; 0 1 0 -1; 0 0 0 0]
    geoms = ["tetrahedral" => hcat(zeros(3), tet), "square planar" => hcat(zeros(3), sq)]
    for (g, X) in geoms
        hh = [norm(X[:, j] - X[:, k]) for j in 2:5 for k in j+1:5]
        println(rpad(g, 13), ": H-H distances ", sort(unique(round.(hh; digits = 3))))
    end
    for rc in (1.5, 5.0)
        diffs = Float64[]
        for seed in 0:2
            m = DenseSchNet([7, 2, 2, 2, 2]; nf = 32, nrbf = 20, rc = rc, ninter = 3)   # tokens = Z + 1
            ps, st = Lux.setup(Xoshiro(seed), m); ps = to64(ps)
            Es = [first(m(reshape(X, 3, 5, 1), ps, st))[1] for (_, X) in geoms]
            push!(diffs, abs(Es[1] - Es[2]))
        end
        @printf("SchNet r_c=%.1f: |E_tet - E_sq| over 3 random inits = %s\n", rc, join([@sprintf("%.1e", d) for d in diffs], ", "))
    end
    println("angular sums at C:  S_l = sum_{j!=k} P_l(cos theta_jk)")
    for (g, X) in geoms
        U = X[:, 2:5] ./ sqrt.(sum(abs2, X[:, 2:5]; dims = 1))
        C = U' * U
        S = [sum(legendre(l, C[j, k]) for j in 1:4, k in 1:4 if j != k) for l in 1:4]
        println("  ", rpad(g, 13), ": ", join([@sprintf("l=%d: %+.4f", l, S[l]) for l in 1:4], "  "))
    end
end

"르장드르 다항식 P_l(x) (3항 점화식; so3.jl에도 같은 함수가 있다)"
function legendre(l, x)
    l == 0 && return one(x)
    p0, p1 = one(x), x
    for k in 2:l
        p0, p1 = p1, ((2k - 1) * x * p1 - (k - 1) * p0) / k
    end
    return p1
end

partA()
R, Z = partB()
partC(R, Z)
partD()
