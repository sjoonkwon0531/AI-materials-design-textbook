"""Ex 5.7.4 (cont.): temperature dependence of the excess (virial) pressure of the rho=3, a=25 DPD fluid,
at small dt (0.01, lambda=0.5), temperature set by sigma^2 = 2 gamma kT."""
import sys, numpy as np
sys.path.insert(0, "../../../ch05/labs")
from dpd_basics import simulate, mean_err
from dpd_core import make_amat
box = (6.0, 6.0, 6.0); n = 648; typ = np.zeros(n, np.int64); A = make_amat(25.0, 0.0)
for kT in (1.0, 1.1, 1.2):
    sig = np.sqrt(2*4.5*kT); Pex = []; Ts = []
    for sd in (3, 4, 5, 6):
        _, _, o = simulate(box, typ, A, 4.5, sig, 0.01, 0.5, 10.0, 60.0, 5, sd)
        Pex.append(mean_err(o[:, 1:].mean(1) - 3*o[:, 0])); Ts.append(o[:, 0].mean())
    m = np.mean([p[0] for p in Pex]); e = np.sqrt(np.sum([p[1]**2 for p in Pex]))/4
    print(f"target kT={kT:.1f} (sigma={sig:.3f}): measured kT={np.mean(Ts):.4f}, excess pressure P-rho*kT = {m:.3f} +/- {e:.3f}")
