"""Ex 5.2.6: 1/nu from dU4/dbeta at the critical point, 2D (L=8..64) and 3D (L=6..32) Ising.
Same Wolff runs (same seeds, lengths, measurement rule) as the Lab FSS scripts, time series kept.
dU4/dbeta = [m2 (<m4 E>-<m4><E>) - 2 m4 (<m2 E>-<m2><E>)] / (3 m2^3); jackknife 20 blocks."""
import sys, time
import numpy as np
sys.path.insert(0, "../ch05/labs")
from isingkern import run_wolff2d_sweeps, run_wolff3d_sweeps
from reweight import rw_moments, observables

def dU4(E, M, N):
    m = np.abs(M) / N
    m2, m4, e = np.mean(m**2), np.mean(m**4), np.mean(E)
    c4 = np.mean(m**4 * E) - m4 * e
    c2 = np.mean(m**2 * E) - m2 * e
    return (m2 * c4 - 2 * m4 * c2) / (3 * m2**3)

def jack(E, M, N, f, nblk=20):
    n = len(E) // nblk * nblk; E = E[:n]; M = M[:n]
    full = f(E, M, N)
    idx = np.arange(n).reshape(nblk, -1)
    reps = np.array([f(E[np.delete(idx, k, 0).ravel()], M[np.delete(idx, k, 0).ravel()], N) for k in range(nblk)])
    return full, np.sqrt((nblk - 1) / nblk * ((reps - reps.mean()) ** 2).sum())

def fit(Ls, y, dy):
    X = np.vstack([np.ones(len(Ls)), np.log(Ls)]).T
    W = np.diag(1 / (dy / y) ** 2)                      # error of ln y = dy/y
    cov = np.linalg.inv(X.T @ W @ X); c = cov @ X.T @ W @ np.log(y)
    chi2 = float(((np.log(y) - X @ c) ** 2 * np.diag(W)).sum())
    return c[1], np.sqrt(cov[1, 1]), chi2 / max(len(Ls) - 2, 1)

t0 = time.time()
res = {}
bc = 0.5 * np.log(1 + np.sqrt(2))
print("=== 2D, beta_c = 0.4406868, Wolff runs as in the 2D FSS lab ===")
runs2 = {8: 200000, 16: 150000, 32: 80000, 64: 30000}
L2, D2, E2 = [], [], []
for L, n in runs2.items():
    E, M = run_wolff2d_sweeps(L, bc, 5000, n, 300 + L)
    E = E.astype(float); M = M.astype(float); N = L * L
    d, e = jack(E, M, N, dU4)
    # cross-check: centred finite difference of the reweighted U4
    h = 1e-4
    U = observables(rw_moments(E, M, N, bc, np.array([bc - h, bc + h])), np.array([bc - h, bc + h]), N)[0]
    print(f"  L={L:3d}: dU4/dbeta = {d:8.4f} +/- {e:.4f}   (reweighting finite difference {(U[1]-U[0])/(2*h):8.4f})")
    L2.append(L); D2.append(d); E2.append(e)
L2, D2, E2 = map(np.array, (L2, D2, E2))
for k in range(len(L2) - 1):
    s, ds, c2 = fit(L2[k:], D2[k:], E2[k:])
    print(f"  fit L >= {L2[k]:2d}: 1/nu = {s:.4f} +/- {ds:.4f}  (chi2/dof = {c2:.2f})")
print(f"  2D time {time.time()-t0:.0f} s")

t1 = time.time()
b0 = 0.2216546
print("=== 3D, beta0 = 0.2216546, Wolff runs as in the 3D FSS lab ===")
runs3 = {6: 60000, 8: 60000, 12: 40000, 16: 30000, 24: 15000, 32: 8000}
L3, D3, E3 = [], [], []
for L, n in runs3.items():
    E, M = run_wolff3d_sweeps(L, b0, 3000, n, 500 + L)
    E = E.astype(float); M = M.astype(float); N = L ** 3
    d, e = jack(E, M, N, dU4)
    print(f"  L={L:3d}: dU4/dbeta = {d:9.4f} +/- {e:.4f}")
    L3.append(L); D3.append(d); E3.append(e)
L3, D3, E3 = map(np.array, (L3, D3, E3))
for k in range(len(L3) - 1):
    s, ds, c2 = fit(L3[k:], D3[k:], E3[k:])
    print(f"  fit L >= {L3[k]:2d} ({len(L3)-k} sizes): 1/nu = {s:.4f} +/- {ds:.4f}  (chi2/dof = {c2:.2f})   bootstrap 1.5874")
print("  successive pairs: " + ", ".join(f"({L3[k]},{L3[k+1]}): {np.log(D3[k+1]/D3[k])/np.log(L3[k+1]/L3[k]):.3f}" for k in range(len(L3)-1)))
print("  2D successive pairs: " + ", ".join(f"({L2[k]},{L2[k+1]}): {np.log(D2[k+1]/D2[k])/np.log(L2[k+1]/L2[k]):.3f}" for k in range(len(L2)-1)))
# fit with a correction term: D = A L^{1/nu} (1 + B L^-omega), omega = 0.83 fixed
from scipy.optimize import curve_fit
f = lambda L, A, x, B: A * L**x * (1 + B * L**-0.83)
p, cv = curve_fit(f, L3, D3, p0=[1, 1.6, 0], sigma=E3, absolute_sigma=True)
print(f"  3D fit with correction (omega=0.83): 1/nu = {p[1]:.3f} +/- {np.sqrt(cv[1,1]):.3f}, B = {p[2]:.2f}")
print(f"  3D time {time.time()-t1:.0f} s")
