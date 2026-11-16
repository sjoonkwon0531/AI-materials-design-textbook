# Ex 2.2.R: embedded error estimate scales as dt^(p+1) = dt^5 for DOPRI5's 4th-order companion
import numpy as np
from dopri5 import step
from kepler import kepler, Y0
hs = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125])
errs = []
for h in hs:
    _, e, _ = step(kepler, 0.0, Y0, h, kepler(0.0, Y0)); errs.append(np.linalg.norm(e))
errs = np.array(errs)
for h, e in zip(hs, errs): print(f"dt={h:<8} |err est|={e:.3e}")
sl = np.diff(np.log(errs))/np.diff(np.log(hs)); print("local slopes d log e / d log dt:", np.round(sl, 3))
# the controller law then hits the tolerance in one shot
tol = 1e-8; h = 0.01; _, e, _ = step(kepler, 0.0, Y0, h, kepler(0.0, Y0)); e = np.linalg.norm(e)
hn = h*(tol/e)**(1/5); _, e2, _ = step(kepler, 0.0, Y0, hn, kepler(0.0, Y0))
print(f"dt=0.01 gives err {e:.3e}; dt_new = dt*(tol/err)^(1/5) = {hn:.5f} gives err {np.linalg.norm(e2):.3e} (tol {tol:g})")
hn3 = h*(tol/e)**(1/3); _, e3, _ = step(kepler, 0.0, Y0, hn3, kepler(0.0, Y0))
print(f"wrong exponent 1/3: dt_new={hn3:.5f} gives err {np.linalg.norm(e3):.3e}")
