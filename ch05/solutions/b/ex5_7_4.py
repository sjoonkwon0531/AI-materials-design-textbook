"""Ex 5.7.4(b): DPD pressure and kinetic temperature vs dt for lambda = 0.65 (and 0.5), rho=3, a=25."""
import sys, time, numpy as np
sys.path.insert(0, "../ch05/labs")
from dpd_basics import simulate, mean_err
from dpd_core import make_amat
t0 = time.time()
box = (6.0, 6.0, 6.0); n = 648; typ = np.zeros(n, np.int64); A = make_amat(25.0, 0.0)
print("(a) sigma=3, gamma=6 -> kT = sigma^2/(2 gamma) =", 9/12)
ref = None
for lam, dts in ((0.5, (0.01,)), (0.65, (0.01, 0.02, 0.04, 0.06, 0.08, 0.09, 0.10))):
    for dt in dts:
        Ps = []; Ts = []
        for sd in (3, 4, 5, 6):
            _, _, o = simulate(box, typ, A, 4.5, 3.0, dt, lam, 10.0, 120.0, 5, sd)
            Ps.append(mean_err(o[:, 1:].mean(1))); Ts.append(mean_err(o[:, 0]))
        P = np.mean([p[0] for p in Ps]); eP = np.sqrt(np.sum([p[1]**2 for p in Ps]))/len(Ps)
        T = np.mean([t[0] for t in Ts]); eT = np.sqrt(np.sum([t[1]**2 for t in Ts]))/len(Ts)
        if ref is None: ref = P
        print(f"lambda={lam:.2f} dt={dt:.2f}: P = {P:.3f} +/- {eP:.3f}  ({(P/23.67-1)*100:+.2f}% vs 23.67)   "
              f"kT = {T:.4f} +/- {eT:.4f}  ({(T-1)*100:+.2f}%)   P - rho*kT = {P-3*T:.3f}", flush=True)
print(f"wall {time.time()-t0:.0f} s")
