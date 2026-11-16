# Lab 4.1 (Julia) — 1D Schrödinger 방정식을 고유값 문제로 풀기 (FDM + 대칭 고유값 솔버)
# 환산단위 hbar = m = 1:  -1/2 psi'' + V psi = E psi  ->  H v = E v
# Python 판(scipy.linalg.eigh)과 같은 격자·같은 행렬. Symmetric(H)로 감싸면 Julia는
# LAPACK 대칭 솔버를 쓰고 고유값을 오름차순으로 돌려준다.
using LinearAlgebra, Printf

"FDM Hamiltonian: H = -1/2 D2 + diag(V), Dirichlet BC (dense, 파이썬 판과 동일)"
function build_hamiltonian(x, V)
    N = length(x); h = x[2] - x[1]                 # 1-based: x[2]-x[1]
    D2 = (diagm(0 => fill(-2.0, N), 1 => ones(N - 1), -1 => ones(N - 1))) ./ h^2
    return -0.5 .* D2 .+ diagm(0 => V)
end

# ---------- Case A: 조화진동자 V = x^2/2, E_n = n + 1/2 ----------
L, N = 10.0, 1000
x = collect(range(-L, L; length = N))              # np.linspace(-L, L, N)
H = build_hamiltonian(x, 0.5 .* x .^ 2)
println("=== Structure check ===")
println("Hamiltonian is symmetric (Hermitian, real): ", issymmetric(H))
F = eigen(Symmetric(H))                            # 오름차순 고유값 + 직교 고유벡터
E, psi = F.values, F.vectors

println("\n=== Case A: Harmonic oscillator  E_n = n + 1/2 ===")
println(" n   E_numeric   E_exact   abs.err")
for n in 0:5
    @printf("%2d  %9.5f  %8.4f  %.2e\n", n, E[n+1], n + 0.5, abs(E[n+1] - (n + 0.5)))
end
overlap = [dot(psi[:, i], psi[:, j]) for i in 1:4, j in 1:4]
println("\nEigenvector overlap matrix <i|j> (should be Identity):")
show(stdout, "text/plain", round.(overlap; digits = 4) .+ 0.0); println()

# ---------- Case B: 무한 우물, L=1, E_n = n^2 pi^2 / (2 L^2) ----------
Lb, Nb = 1.0, 2000
xb = collect(range(0, Lb; length = Nb))
Eb = eigvals(Symmetric(build_hamiltonian(xb, zeros(Nb))))
println("\n=== Case B: Infinite square well  E_n = n^2 pi^2 / (2 L^2),  L=1 ===")
println(" n   E_numeric   E_exact    rel.err")
for n in 1:5
    Eex = n^2 * pi^2 / (2 * Lb^2)
    @printf("%2d  %9.4f  %9.4f  %.2e\n", n, Eb[n], Eex, abs(Eb[n] - Eex) / Eex)
end

# ---------- 격자 수렴: 바닥상태 오차 vs N (O(h^2)) ----------
print("\nground-state grid-convergence errors: \n")
for Ng in (100, 200, 400, 800, 1600, 3200)
    xg = collect(range(-L, L; length = Ng))
    Eg = eigvals(Symmetric(build_hamiltonian(xg, 0.5 .* xg .^ 2)))
    @printf("%s%d:%.2e  ", Ng == 100 ? "N=" : "", Ng, abs(Eg[1] - 0.5))
end
println()

# ---------- Case C: 변분원리, Gauss 시험함수 E(a) = a/4 + 1/(4a) ----------
println("\n=== Case C: Variational principle, trial Gaussian for HO ===")
println("E(a) = a/4 + 1/(4a),  exact E0 = 0.5 at a=1")
println("   a     E_trial(a)")
Etrial(a) = a / 4 + 1 / (4a)
for a in (0.4, 0.7, 1.0, 1.5, 2.5)
    @printf("  %4.2f   %8.5f\n", a, Etrial(a))
end
avals = range(0.2, 4.0; length = 400)
Evar = Etrial.(avals)
imin = argmin(Evar)
@printf("numerical minimum at a = %.4f, E = %.5f  (>= 0.5, equality at a=1)\n", avals[imin], Evar[imin])
