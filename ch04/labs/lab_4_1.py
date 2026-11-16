import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# Lab 4.1  Numerical solution of the 1D time-independent
#          Schrodinger equation by finite differences.
#   Reduced (atomic-like) units: hbar = m = 1.
#   -1/2 psi'' + V(x) psi = E psi  ->  H psi = E psi
# ============================================================

def build_hamiltonian(x, V):
    """Build FDM Hamiltonian on grid x with potential array V."""
    N  = len(x)
    h  = x[1] - x[0]
    main = np.full(N, -2.0)
    off  = np.full(N-1, 1.0)
    D2   = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / h**2
    T    = -0.5 * D2
    H    = T + np.diag(V)
    return H

L  = 10.0
N  = 1000
x  = np.linspace(-L, L, N)
V_ho = 0.5 * x**2
H_ho = build_hamiltonian(x, V_ho)

print("=== Structure check ===")
print("Hamiltonian is symmetric (Hermitian, real):",
      np.allclose(H_ho, H_ho.T))

E_ho, psi_ho = la.eigh(H_ho)

print("\n=== Case A: Harmonic oscillator  E_n = n + 1/2 ===")
print(" n   E_numeric   E_exact   abs.err")
for n in range(6):
    print(f"{n:2d}  {E_ho[n]:9.5f}  {n+0.5:8.4f}  {abs(E_ho[n]-(n+0.5)):.2e}")

dx = x[1]-x[0]
overlap = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        overlap[i,j] = np.sum(psi_ho[:,i]*psi_ho[:,j])
print("\nEigenvector overlap matrix <i|j> (should be Identity):")
print(overlap)

Lb = 1.0
Nb = 2000
xb = np.linspace(0, Lb, Nb)
V_box = np.zeros(Nb)
H_box = build_hamiltonian(xb, V_box)
E_box, psi_box = la.eigh(H_box)
print("\n=== Case B: Infinite square well  E_n = n^2 pi^2 / (2 L^2),  L=1 ===")
print(" n   E_numeric   E_exact    rel.err")
for n in range(1,6):
    Eex = (n**2 * np.pi**2) / (2*Lb**2)
    print(f"{n:2d}  {E_box[n-1]:9.4f}  {Eex:9.4f}  {abs(E_box[n-1]-Eex)/Eex:.2e}")

print("\n=== Case C: Variational principle, trial Gaussian for HO ===")
print("E(a) = a/4 + 1/(4a),  exact E0 = 0.5 at a=1")
print("   a     E_trial(a)")
for a in [0.4, 0.7, 1.0, 1.5, 2.5]:
    Etrial = a/4 + 1.0/(4*a)
    print(f"  {a:4.2f}   {Etrial:8.5f}")
avals = np.linspace(0.2, 4.0, 400)
Evar  = avals/4 + 1.0/(4*avals)
amin  = avals[np.argmin(Evar)]
print(f"numerical minimum at a = {amin:.4f}, E = {Evar.min():.5f}  (>= 0.5, equality at a=1)")

Ns = [100, 200, 400, 800, 1600, 3200]
errs = []
for Ng in Ns:
    xg = np.linspace(-L, L, Ng)
    Hg = build_hamiltonian(xg, 0.5*xg**2)
    Eg = la.eigvalsh(Hg)
    errs.append(abs(Eg[0]-0.5))
print("ground-state grid-convergence errors:", [f'{e:.2e}' for e in errs])
