"""Ex 5.1.5: Markov-closure error vs g/a for the 2-variable linear system of Derivation Box 5.2
(a=b=1, c=-1, x0=1, y0=0, t in [0,6], 601 points as in the lab), full window vs t >= 5/g,
and a decomposition into initial slip (amplitude renormalisation) and rate error."""
import numpy as np
from scipy.linalg import expm
a, b, c = 1.0, 1.0, -1.0
gs = np.array([1, 2, 5, 10, 20, 50, 100], float)
tt = np.linspace(0, 6, 601)
E_full, E_late, E_rate, E_amp, dlam, Af = [], [], [], [], [], []
print("  g/a   max_all     max_t>=5/g   lam_s       lam_M      dlam        A_s-1      rate-only  amp-only")
for g in gs:
    A = np.array([[-a, b], [c, -g]])
    t_fine = np.union1d(tt, np.linspace(5 / g, 6, 2001))     # include t0 = 5/g exactly
    xe = np.array([(expm(A * t) @ [1.0, 0.0])[0] for t in t_fine])
    lamM = -a + b * c / g
    xm = np.exp(lamM * t_fine)
    ev = np.linalg.eigvals(A)
    if np.abs(ev.imag).max() > 0 or abs(ev[0] - ev[1]) < 1e-9:   # (g-a)^2 + 4bc <= 0: no slow/fast split
        ls = lf = As = np.nan
    else:
        ev = np.sort(ev.real); ls, lf = ev[1], ev[0]                 # ev[1] slow, ev[0] fast
        As = (-a - lf) / (ls - lf)                    # x = As e^{ls t} + Af e^{lf t}, xdot(0) = -a
    on = np.isin(t_fine, tt)
    e_all = np.abs(xe - xm)[on].max()
    e_late = np.abs(xe - xm)[t_fine >= 5 / g - 1e-12].max()
    e_rate = np.abs(As * np.exp(ls * t_fine) - As * np.exp(lamM * t_fine)).max()   # same amplitude, wrong rate
    e_amp = np.abs(As * np.exp(ls * t_fine) - np.exp(ls * t_fine)).max()           # amplitude offset only
    E_full.append(e_all); E_late.append(e_late); E_rate.append(e_rate); E_amp.append(e_amp)
    print(f"{g:5.0f}  {e_all:.3e}   {e_late:.3e}   {ls:.6f}  {lamM:.6f}  {ls-lamM:+.3e}  {As-1:+.3e}  {e_rate:.2e}   {e_amp:.2e}")
lg = np.log(gs)
for name, E in [("all t", E_full), ("t>=5/g", E_late), ("rate-only", E_rate), ("amplitude-only", E_amp)]:
    E = np.array(E); ok = np.isfinite(E)
    s_all = np.polyfit(lg[ok], np.log(E[ok]), 1)[0] if ok.all() else np.nan
    s_hi = np.polyfit(lg[3:], np.log(E[3:]), 1)[0]
    print(f"slope {name:15s}: all g (1-100) = {s_all:.2f}; g/a >= 10: {s_hi:.2f}; g/a 5-100: {np.polyfit(lg[2:], np.log(E[2:]), 1)[0]:.2f}")
# where is the maximum error located?
for g in (10.0, 100.0):
    A = np.array([[-a, b], [c, -g]])
    xe = np.array([(expm(A * t) @ [1.0, 0.0])[0] for t in tt])
    xm = np.exp((-a + b * c / g) * tt)
    print(f"g={g:.0f}: argmax_t |err| = {tt[np.argmax(np.abs(xe-xm))]:.2f}  (5/g = {5/g:.2f})")
# Markov with corrected initial amplitude and corrected rate (second-order closure)
g = 10.0
A = np.array([[-a, b], [c, -g]]); ev = np.sort(np.linalg.eigvals(A).real); ls, lf = ev[1], ev[0]
As = (-a - lf) / (ls - lf)
xe = np.array([(expm(A * t) @ [1.0, 0.0])[0] for t in tt])
lam2 = -a + b * c / g + a * b * c / g**2
print(f"g=10: x0_eff=A_s={As:.6f}; closure with A_s and lam2=-a+bc/g+abc/g^2={lam2:.6f} (exact {ls:.6f}): "
      f"max err all t = {np.abs(xe - As*np.exp(lam2*tt)).max():.2e}, t>=5/g = {np.abs(xe - As*np.exp(lam2*tt))[tt>=0.5].max():.2e}")
