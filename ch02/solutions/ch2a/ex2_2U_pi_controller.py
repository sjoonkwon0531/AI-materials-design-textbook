# Ex 2.2.U: elementary (I) vs PI step controller in DOPRI5
import numpy as np
from dopri5 import integrate
from kepler import kepler, Y0, T4
from stiffnet import A, X0
def metrics(st, skip=5):
    lh = np.log(st['hs'][skip:-2]); d = np.diff(lh)
    return np.mean(np.abs(d)), np.mean(np.sign(d[1:]) != np.sign(d[:-1]))
cases = [("Kepler e=0.8, rtol=1e-6", kepler, Y0, T4, 1e-6, 1e-9),
         ("stiff network (Lab 2.3), t<=8, rtol=1e-6", lambda t, X: A@X, X0, 8.0, 1e-6, 1e-9)]
ctrls = [("I  (err^-1/5)", dict(controller="I")),
         ("PI (kI=0.3/5,kP=0.4/5)", dict(controller="PI")),
         ("PI (kI=0.6/5,kP=0.2/5)", dict(controller="PI", kI=0.6/5, kP=0.2/5))]
for name, f, y0, T, rtol, atol in cases:
    print(name)
    for cn, kw in ctrls:
        t, y, st = integrate(f, 0.0, y0, T, rtol, atol, 1e-4, **kw)
        md, sc = metrics(st)
        print(f"  {cn:24s} acc={st['accepted']:5d} rej={st['rejected']:4d} nfev={st['nfev']:6d}  mean|dlog h|={md:.4f}  sign-change frac={sc:.2f}")
# stability-limited regime detail: h in the slow phase t>0.5 of the stiff network
import numpy as np
for cn, kw in ctrls[:2]:
    t, y, st = integrate(lambda t, X: A@X, 0.0, X0, 8.0, 1e-6, 1e-9, 1e-4, **kw)
    hs = st['hs']; tt = np.cumsum(hs); m = tt > 0.5
    print(f"  stiff, t>0.5, {cn:24s}: mean h={hs[m].mean():.5f} (stab. bound ~3.3/1001.5={3.3/1001.5:.5f}), std/mean={hs[m].std()/hs[m].mean():.3f}")
# DOPRI5 real-axis stability bound from its tableau: R(z) = 1 + z b^T (I - zA)^{-1} 1 (5th-order weights)
from scipy.optimize import brentq
from dopri5 import Am, B5
Af = np.zeros((7, 7))
for i in range(7):
    for j, a in enumerate(Am[i]): Af[i, j] = a
R = lambda z: 1 + z*B5 @ np.linalg.solve(np.eye(7) - z*Af, np.ones(7))
xr = brentq(lambda x: abs(R(x)) - 1, -4.0, -2.9)
print(f"DOPRI5 real-axis stability interval [{xr:.4f}, 0] -> h_max = {abs(xr)/1001.5:.5f} for lam=-1001.5")
