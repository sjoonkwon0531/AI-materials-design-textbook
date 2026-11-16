# Ex 2.2.N: stiffness ratio and explicit-Euler step limit for the Lab 2.3 matrix
import numpy as np
from scipy.linalg import expm
from stiffnet import A, X0, T
lam = np.sort(np.linalg.eigvals(A).real)
print("A =\n", A); print("column sums:", A.sum(axis=0)); print("eigenvalues:", lam)
nz = lam[np.abs(lam) > 1e-12]
S = np.abs(nz).max()/np.abs(nz).min(); hmax = 2/np.abs(nz).max()
print(f"S = {S:.2f}, explicit Euler limit dt <= 2/|lam_max| = {hmax:.6f}")
for h in [0.0019, 0.00199, 0.00200, 0.0021]:
    n = int(round(T/h)); x = X0.copy(); M = np.eye(3) + h*A
    for _ in range(n): x = M@x
    print(f"  dt={h}: |g_fast|={abs(1+h*nz.min()):.4f}, max|X(8)-exact|={np.abs(x-expm(A*n*h)@X0).max():.2e}")
