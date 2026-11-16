"""Ex 5.4.4: size of the pressure-correction ramp. Analytic estimate vs actual CG runs that separate
the ramp from the simultaneous IBI update (Lab step 'pressure-corrected IBI', iteration 0 -> 1)."""
import numpy as np
from ibi_chain import get_chain, ibi, NMOL, T, DT, L

c = get_chain()
r = ibi.RGRID; dr = ibi.DR; rho = NMOL / L**2; rc = ibi.RC_CG
g_t, V, valid, P_fine = c["g_t"], c["V"], c["valid"], float(c["P_fine"])
P0, g0 = float(c["P_p0"]), c["g_p0"]
dP = P0 - P_fine
A = -0.1 * T * np.sign(dP) * min(1.0, 3.0 * abs(dP))
print(f"rho={rho:.4f}, P_p0={P0:.4f}, P_fine={P_fine:.4f}, dP={dP:.4f}, A={A:.5f}")
# (a)/(b) analytic: dP = pi rho^2 A/(2 rc) int_0^rc r^2 g dr
I1 = rc**3 / 3; Ig = np.sum(r**2 * g_t) * dr
print(f"int r^2 dr = {I1:.4f};  int r^2 g_target dr = {Ig:.4f} (ratio {Ig/I1:.4f})")
print(f"(b) g=1: dP = pi rho^2 A rc^2/6 = {np.pi*rho**2*A*rc**2/6:.5f};  with g_target: {np.pi*rho**2*A/(2*rc)*Ig:.5f}")
print(f"    contribution of r<1 to int r^2 dr: {1/3:.4f}; to int r^2 g dr: {np.sum((r**2*g_t)[r<1])*dr:.4f}; "
      f"excess of int r^2 (g-1) dr over 1<r<2: {np.sum((r**2*(g_t-1))[(r>1)&(r<2)])*dr:.4f}")
# the potentials actually used: ramp only, IBI only, both (Lab), each through finalize()
m = valid & (g0 > 1e-3)
def fin(Vx): return ibi.finalize(Vx, valid)
V_ramp, F_ramp = fin(V + A * (1 - r / rc))
Vi = V.copy(); Vi[m] += T * np.log(g0[m] / g_t[m]); V_ibi, F_ibi = fin(Vi)
Vb = Vi + A * (1 - r / rc); V_both, F_both = fin(Vb)
V_none, F_none = fin(V.copy())
def lin_dP(Fnew, Fold, g):     # virial linear response with g fixed: dP = (rho^2/2) * pi * int r^2 dF g dr  (F = -V')
    return np.pi * rho**2 / 2 * np.sum(r**2 * (Fnew - Fold) * g) * dr
print(f"linear response with fixed g_target, finalized potentials: ramp only {lin_dP(F_ramp, F_none, g_t):.5f}, "
      f"IBI only {lin_dP(F_ibi, F_none, g_t):.5f}, both {lin_dP(F_both, F_none, g_t):.5f}")
print(f"   (finalize() vs raw: max |V_none - V| = {np.abs(V_none - V).max():.2e})")
res = {}
for name, F in [("unchanged", F_none), ("ramp only", F_ramp), ("IBI only", F_ibi), ("ramp+IBI (Lab)", F_both)]:
    Ps = []
    for sd in (301, 401, 501, 601):
        hc, nf, P, _ = ibi.run_cg(NMOL, L, F, 5000, 30000, DT, 1.0, sd, 15, 2.0)
        Ps.append(P)
    Ps = np.array(Ps); res[name] = Ps
    print(f"  {name:15s}: P_CG per seed (301,401,501,601) = {np.round(Ps,4)}, mean = {Ps.mean():.4f} +/- {Ps.std(ddof=1)/2:.4f} (sd {Ps.std(ddof=1):.4f})", flush=True)
b = res["unchanged"].mean()
for k in ("ramp only", "IBI only", "ramp+IBI (Lab)"):
    print(f"  mean change vs unchanged: {k:15s} {res[k].mean()-b:+.4f}")
