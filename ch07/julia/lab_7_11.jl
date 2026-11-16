# lab_7_11.jl — Lab 7.11: 군을 숫자로 만지기 (labs/73a_group.py)
# Part A 생성원에서 점군 닫기(C3v, D4h, Oh, Td), 공리 점검. Part B SO(2), O(3). Part C C3v 지표와 기약 표현 분해.
# Part D E(3) 원소 (R, t)의 합성 규칙과 역원. 순수 선형대수라 그대로 옮겼다.
include("so3.jl")
using Printf

"생성원을 포함하고 곱에 닫힌 가장 작은 행렬 집합"
function close_group(gens; tol = 1e-9)
    G = [Matrix(1.0I, 3, 3)]; frontier = copy(G)
    while !isempty(frontier)
        new = Matrix{Float64}[]
        for A in frontier, g in gens
            B = g * A
            if !any(maximum(abs.(B - C)) < tol for C in G)
                push!(G, B); push!(new, B)
            end
        end
        frontier = new
    end
    return G
end
isin(M, G; tol = 1e-9) = any(maximum(abs.(M - C)) < tol for C in G)

function main()
    rng = Xoshiro(0)
    println("=== Part A: point groups from generators")
    groups = ["C3v (NH3)" => [rot([0, 0, 1], 2pi / 3), mirror([0, 1, 0])],
              "D4h (square planar)" => [rot([0, 0, 1], pi / 2), rot([1, 0, 0], pi), -Matrix(1.0I, 3, 3)],
              "Oh (cube, SF6)" => [rot([0, 0, 1], pi / 2), rot([1, 1, 1], 2pi / 3), -Matrix(1.0I, 3, 3)],
              "Td (CH4)" => [rot([1, 1, 1], 2pi / 3), mirror([1, -1, 0]), rot([0, 0, 1], pi)]]
    for (name, gens) in groups
        G = close_group(gens); n = length(G)
        closure = all(isin(A * B, G) for A in G, B in G)
        inverses = all(any(maximum(abs.(A * B - I)) < 1e-9 for B in G) for A in G)
        assoc = maximum(maximum(abs.((G[i] * G[j]) * G[k] - G[i] * (G[j] * G[k]))) for (i, j, k) in eachrow(rand(rng, 1:n, 200, 3)))
        abel = all(maximum(abs.(A * B - B * A)) < 1e-9 for A in G, B in G)
        dp = count(A -> det(A) > 0, G)
        @printf("%-26s order %2d | closure %s | inverses %s | assoc. err %.1e | abelian %s | det(+1):%d det(-1):%d\n",
                name, n, closure, inverses, assoc, abel, dp, n - dp)
    end

    println("=== Part B: continuous groups")
    R2(t) = [cos(t) -sin(t); sin(t) cos(t)]
    th = 2pi .* rand(rng, 1000, 2)
    @printf("SO(2): max |D(a)D(b) - D(a+b)| over 1000 pairs = %.1e\n", maximum(maximum(abs.(R2(a) * R2(b) - R2(a + b))) for (a, b) in eachrow(th)))
    Rx, Rz = rot([1, 0, 0], pi / 2), rot([0, 0, 1], pi / 2)
    @printf("SO(3): |Rx(90)Rz(90) - Rz(90)Rx(90)|_max = %.3f -> non-abelian\n", maximum(abs.(Rx * Rz - Rz * Rx)))
    Q = Matrix(qr(randn(rng, 3, 3)).Q)
    @printf("random orthogonal Q: |Q'Q - I| = %.1e, det = %+.6f\n", maximum(abs.(Q' * Q - I)), det(Q))

    println("=== Part C: characters of C3v (NH3)")
    G = close_group(groups[1][2])
    H = [0.94 * cos(a) for a in (0, 2pi / 3, 4pi / 3)]'
    H = vcat(H, [0.94 * sin(a) for a in (0, 2pi / 3, 4pi / 3)]', fill(-0.35, 1, 3))   # 3 x 3, 열 = H 원자
    cls(A) = maximum(abs.(A - I)) < 1e-9 ? "E" : det(A) > 0 ? "2C3" : "3sv"
    classes = ["E", "2C3", "3sv"]; sizes = Dict("E" => 1, "2C3" => 2, "3sv" => 3)
    irreps = Dict("A1" => Dict("E" => 1, "2C3" => 1, "3sv" => 1), "A2" => Dict("E" => 1, "2C3" => 1, "3sv" => -1),
                  "E" => Dict("E" => 2, "2C3" => -1, "3sv" => 0))
    function perm_matrix(A)                         # H 원자 순열 표현: A H_i = H_j 이면 P[j, i] = 1
        P = zeros(3, 3)
        for i in 1:3
            j = argmin([norm(H[:, k] - A * H[:, i]) for k in 1:3]); P[j, i] = 1
        end
        return P
    end
    for (rname, f) in ("vector (x,y,z)" => identity, "H permutations" => perm_matrix)
        chi = Dict(c => mean(tr(f(A)) for A in G if cls(A) == c) for c in classes)
        mult = Dict(ir => sum(sizes[c] * chi[c] * irreps[ir][c] for c in classes) / length(G) for ir in keys(irreps))
        hom = maximum(maximum(abs.(f(A * B) - f(A) * f(B))) for A in G, B in G)
        @printf("%-16s: chi = %s -> A1:%.0f, A2:%.0f, E:%.0f | homomorphism err %.1e\n", rname,
                [round(chi[c]; digits = 6) for c in classes], mult["A1"], mult["A2"], mult["E"], hom)
    end

    println("=== Part D: E(3) elements g = (R, t) acting as x -> R x + t (X: 3 x 9, 열 = 원자)")
    act(g, X) = g[1] * X .+ g[2]
    compose(g, h) = (g[1] * h[1], g[1] * h[2] + g[2])
    inv_(g) = (g[1]', -g[1]' * g[2])
    X = randn(rng, 3, 9)
    g = (rot(randn(rng, 3), 1.0) * mirror([0, 0, 1]), randn(rng, 3))
    h = (rot(randn(rng, 3), 2.0), randn(rng, 3))
    @printf("|g.(h.X) - (gh).X| = %.1e,  |g^-1.(g.X) - X| = %.1e\n", maximum(abs.(act(g, act(h, X)) - act(compose(g, h), X))),
            maximum(abs.(act(inv_(g), act(g, X)) - X)))
    @printf("translations do not commute with rotations: |(gh).t - (hg).t| = %.3f\n", maximum(abs.(compose(g, h)[2] - compose(h, g)[2])))
    D(Y) = [norm(Y[:, i] - Y[:, j]) for i in 1:9, j in 1:9]
    @printf("all pair distances preserved by g (det R = %+.0f): max change %.1e\n", det(g[1]), maximum(abs.(D(act(g, X)) - D(X))))
end

main()
