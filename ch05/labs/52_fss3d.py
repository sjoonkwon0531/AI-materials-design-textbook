"""Lab 5.2 (key 52_fss), part 2: 3D Ising T_c from Binder crossings with corrections to scaling.
Wolff runs (one per L) at beta0 = 0.2216546, single-histogram reweighting of U4(T),
crossings T*(L, 2L), extrapolation T*(L) = T_c + a L^-(omega + 1/nu) with omega = 0.83, 1/nu = 1.587.
Runtime about 4-5 min (single thread). Seeds fixed; jackknife errors (20 blocks).
"""
import time
import numpy as np
from isingkern import run_wolff3d_sweeps
from reweight import rw_moments, observables, jackknife

t0 = time.time()
b0 = 0.2216546
runs = {6: 60000, 8: 60000, 12: 40000, 16: 30000, 24: 15000, 32: 8000}
T = np.linspace(4.44, 4.58, 281)
betas = 1 / T
data = {}
print(f"=== 3D Ising (simple cubic), Wolff, one run per L at beta0 = {b0} (T0 = {1/b0:.4f}) ===")
for L, n in runs.items():
    t = time.time()
    E, M = run_wolff3d_sweeps(L, b0, 3000, n, 500 + L)
    data[L] = (E, M)
    N = L ** 3
    f = lambda mom: observables(mom, np.array([b0]), N)[0]
    u, du, _ = jackknife(E, M, N, b0, np.array([b0]), func=f)
    print(f"  L={L:3d}: {n:6d} sweeps, U4(T0) = {u[0]:.4f} +/- {du[0]:.4f}, sigma_E*|dbeta| at T=4.44/4.58: "
          f"{np.std(E)*abs(1/4.44-b0):.1f}/{np.std(E)*abs(1/4.58-b0):.1f}   ({time.time()-t:.0f} s)")

curves, reps = {}, {}
for L, (E, M) in data.items():
    N = L ** 3
    f = lambda mom: observables(mom, betas, N)[0]
    curves[L], _, reps[L] = jackknife(E, M, N, b0, betas, func=f)


def crossing(T, ua, ub):
    d = ua - ub
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(idx) == 0:
        return np.nan
    k = idx[np.argmin(np.abs(T[idx] - 1 / b0))]
    return T[k] - d[k] * (T[k + 1] - T[k]) / (d[k + 1] - d[k])


pairs = [(6, 12), (8, 16), (12, 24), (16, 32)]
print("=== Binder crossings T*(L, 2L) ===")
res = []
for a, b in pairs:
    c = crossing(T, curves[a], curves[b])
    cj = np.array([crossing(T, reps[a][k], reps[b][k]) for k in range(len(reps[a]))])
    err = np.sqrt((len(cj) - 1) / len(cj) * np.sum((cj - cj.mean()) ** 2))
    ustar = np.interp(c, T, curves[b])
    res.append((a, c, err, ustar))
    print(f"  ({a:2d},{b:2d}): T* = {c:.4f} +/- {err:.4f}   U* = {ustar:.4f}")

# corrections-to-scaling extrapolation, exponent fixed to omega + 1/nu
w_exp = 0.83 + 1 / 0.630
La = np.array([r[0] for r in res], float)
Ts = np.array([r[1] for r in res])
eT = np.array([r[2] for r in res])
X = np.vstack([np.ones_like(La), La ** (-w_exp)]).T
W = np.diag(1 / eT ** 2)
cov = np.linalg.inv(X.T @ W @ X)
coef = cov @ X.T @ W @ Ts
chi2 = float(((Ts - X @ coef) ** 2 / eT ** 2).sum())
print(f"=== weighted fit T*(L) = T_c + a L^-{w_exp:.2f} over L = {La.astype(int).tolist()} ===")
print(f"  T_c = {coef[0]:.4f} +/- {np.sqrt(cov[0,0]):.4f},  a = {coef[1]:.3f},  chi2/dof = {chi2/(len(La)-2):.2f}")
print(f"  naive (largest pair only): T* = {Ts[-1]:.4f};  reference T_c = 1/0.221654626 = {1/0.221654626:.4f}")
print(f"total {time.time()-t0:.0f} s")
np.savez("../../ch05/labs/52_fss3d_results.npz", T=T, **{f"U_{L}": curves[L] for L in runs},
         La=La, Ts=Ts, eT=eT, coef=coef, w_exp=w_exp)
