"""Exercise 9.6.6 (b), diagnostics: why does the binarised volume fraction differ even when the mass is conserved?"""
import numpy as np, ch_common as cc
homog = cc.load("homog", "../../../ch05/labs/55_homog.py"); cell = homog.PeriodicCell(64)
test = cc.test; ref = test[:, 10]
kfun = lambda fields: np.array([np.trace(homog.k_eff(cell, c > 0, 1.0, 10.0)) / 2 for c in fields])
k_s = kfun(ref); f_s = (ref > 0).mean((1, 2))
def phase_stats(c):
    return np.array([x[x > 0].mean() for x in c]).mean(), np.array([x[x < 0].mean() for x in c]).mean()
print("t = 25                mean |d<c>|   <c | c>0>   <c | c<0>   |df| binarised   median |dk|/k   share of trajectories with |dk|/k > 5%")
print("solver                0             %.4f      %.4f" % phase_stats(ref))
for tag in ("cons", "free"):
    r = cc.rollout(cc.lab_model(tag), test[:, 1], 9)[:, -1]
    k = kfun(r); dk = (k - k_s) / k_s; df = (r > 0).mean((1, 2)) - f_s
    cp, cm = phase_stats(r)
    print(f"{tag:20s}  {np.abs(r.mean((1, 2)) - test[:, 1].mean((1, 2))).mean():.2e}      {cp:.4f}      {cm:.4f}      {np.abs(df).mean():.4f}           {100 * np.median(np.abs(dk)):.2f}%          {int((np.abs(dk) > 0.05).sum())} / 40")
    i = np.argmax(np.abs(dk))
    print(f"    worst trajectory {i}: m = {cc.m_test[i]:+.3f}, f solver {f_s[i]:.3f} / model {(r[i] > 0).mean():.3f}, k_eff solver {k_s[i]:.3f} / model {k[i]:.3f} ({100 * dk[i]:+.1f}%)")
    Ks, Km = homog.k_eff(cell, ref[i] > 0, 1.0, 10.0), homog.k_eff(cell, r[i] > 0, 1.0, 10.0)
    print(f"        k_xx, k_yy: solver {Ks[0, 0]:.3f}, {Ks[1, 1]:.3f}; model {Km[0, 0]:.3f}, {Km[1, 1]:.3f}")
    near = np.abs(f_s - 0.5) < 0.03
    print(f"    mean |dk|/k for |f - 0.5| < 0.03 ({near.sum()} traj.): {100 * np.abs(dk)[near].mean():.2f}%;  otherwise: {100 * np.abs(dk)[~near].mean():.2f}%")
