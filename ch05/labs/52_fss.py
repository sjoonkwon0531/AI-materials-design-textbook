"""Lab 5.2 (key 52_fss), part 1: 2D Ising finite-size scaling with Wolff + histogram reweighting.
One Wolff run per L at the exact beta_c; single-histogram reweighting gives continuous curves
U4(T), chi'(T). Binder crossings, data collapse, and a check of reweighting against direct runs.
Runtime about 2-3 min (single thread). Seeds fixed; jackknife errors (20 blocks).
"""
import time
import numpy as np
from isingkern import run_wolff2d_sweeps
from reweight import rw_moments, observables, jackknife

t0 = time.time()
bc = 0.5 * np.log(1 + np.sqrt(2))
Tc = 1 / bc
runs = {8: 200000, 16: 150000, 32: 80000, 64: 30000}
T = np.linspace(2.0, 2.6, 301)
betas = 1 / T
data = {}
print(f"=== 2D Ising, Wolff (measure once per sweep-equivalent), single run per L at T0 = T_c = {Tc:.6f} ===")
for L, n in runs.items():
    t = time.time()
    E, M = run_wolff2d_sweeps(L, bc, 5000, n, 300 + L)
    data[L] = (E, M)
    N = L * L
    U = lambda mom: observables(mom, np.array([bc]), N)[0]
    u0, du0, _ = jackknife(E, M, N, bc, np.array([bc]), func=U)
    print(f"  L={L:3d}: {n:6d} sweeps, U4(T_c) = {u0[0]:.4f} +/- {du0[0]:.4f}   ({time.time()-t:.0f} s)")


def crossing(T, ua, ub):
    d = ua - ub
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    # choose the sign change closest to T_c
    if len(idx) == 0:
        return np.nan
    k = idx[np.argmin(np.abs(T[idx] - Tc))]
    return T[k] - d[k] * (T[k + 1] - T[k]) / (d[k + 1] - d[k])


# the stopping-rule pitfall (same L=16, same number of measurements)
Eb, Mb = run_wolff2d_sweeps(16, bc, 5000, 150000, 316, True)
fb = lambda mom: observables(mom, np.array([bc]), 256)[0]
ub, dub, _ = jackknife(Eb, Mb, 256, bc, np.array([bc]), func=fb)
print(f"  pitfall, L=16 measured 'after >= N flipped spins' (state-dependent stopping): U4(T_c) = {ub[0]:.4f} +/- {dub[0]:.4f}  (biased)")

# reweighted U4 curves with jackknife replicates
curves, reps = {}, {}
for L, (E, M) in data.items():
    N = L * L
    f = lambda mom: observables(mom, betas, N)[0]
    curves[L], _, reps[L] = jackknife(E, M, N, bc, betas, func=f)

print("=== Binder crossings of reweighted U4(T) (jackknife errors) ===")
Ls = list(runs)
cross = []
for a, b in zip(Ls[:-1], Ls[1:]):
    c = crossing(T, curves[a], curves[b])
    cj = np.array([crossing(T, reps[a][k], reps[b][k]) for k in range(len(reps[a]))])
    err = np.sqrt((len(cj) - 1) / len(cj) * np.sum((cj - cj.mean()) ** 2))
    ustar = np.interp(c, T, curves[b])
    cross.append((a, b, c, err, ustar))
    print(f"  ({a:2d},{b:3d}): T* = {c:.5f} +/- {err:.5f}   (T*-T_c)/T_c = {(c-Tc)/Tc:+.1e}   U* = {ustar:.4f}")
print(f"  exact T_c = {Tc:.5f}")

# reweighting vs direct simulation for L = 32 at two temperatures, plus a failure far outside the window
print("=== check: reweighted (from T_c run) vs direct Wolff run, L=32 ===")
L = 32
N = L * L
E, M = data[L]
sigE = np.std(E)
print(f"  energy std at T_c: sigma_E = {sigE:.1f}  -> window |dbeta| ~ 1/sigma_E = {1/sigE:.4f}  (|dT| ~ {Tc**2/sigE:.3f})")
checks = []
for Td in [2.23, 2.31, 2.60]:
    bd = 1 / Td
    rw = observables(rw_moments(E, M, N, bc, np.array([bd])), np.array([bd]), N)[0][0]
    Ed, Md = run_wolff2d_sweeps(L, bd, 5000, 40000, 900 + int(100 * Td))
    dr = observables(rw_moments(Ed, Md, N, bd, np.array([bd])), np.array([bd]), N)[0][0]
    f = lambda mom: observables(mom, np.array([bd]), N)[0]
    _, edr, _ = jackknife(Ed, Md, N, bd, np.array([bd]), func=f)
    _, erw, _ = jackknife(E, M, N, bc, np.array([bd]), func=f)
    checks.append((Td, rw, erw[0], dr, edr[0]))
    print(f"  T={Td:.2f} (|dbeta|*sigma_E = {abs(bd-bc)*sigE:4.1f}): reweighted U4 = {rw:.4f} +/- {erw[0]:.4f}   direct U4 = {dr:.4f} +/- {edr[0]:.4f}")

# data collapse of chi' with exact exponents
print("=== data collapse: chi' L^(-gamma/nu) vs t L^(1/nu), gamma = 7/4, nu = 1 ===")
coll = {}
for L, (E, M) in data.items():
    N = L * L
    mom = rw_moments(E, M, N, bc, betas)
    U4, chi, chip = observables(mom, betas, N)
    x = (T - Tc) / Tc * L
    win = np.abs(1 / T - bc) * np.std(E) < 2.0         # keep only the reliable reweighting window
    coll[L] = (x[win], (chip * L ** (-1.75))[win], (U4)[win])
    k = np.argmax(chip)
    print(f"  L={L:3d}: chi'_max = {chip[k]:8.3f} at T = {T[k]:.4f}  (x = {x[k]:.2f}),  chi'_max L^-7/4 = {chip[k]*L**-1.75:.4f}")
gn = np.polyfit(np.log(Ls), np.log([np.max(observables(rw_moments(*data[L], L*L, bc, betas), betas, L*L)[2]) for L in Ls]), 1)[0]
print(f"  fit chi'_max ~ L^{gn:.3f}  (gamma/nu = 1.75)")
print(f"total {time.time()-t0:.0f} s")

np.savez("../../ch05/labs/52_fss2d_results.npz", T=T, Tc=Tc,
         **{f"U_{L}": curves[L] for L in Ls},
         **{f"cx_{L}": coll[L][0] for L in Ls}, **{f"cy_{L}": coll[L][1] for L in Ls},
         checks=np.array(checks), cross=np.array(cross))
