# Ex 2.2.T: count rejected steps vs tolerance (own DOPRI5 with scipy-like elementary controller)
import numpy as np
from scipy.integrate import solve_ivp
from dopri5 import integrate
from kepler import kepler, Y0, T4
print("Kepler e=0.8, 4 periods, atol = rtol*1e-3, h0=1e-3")
print(" rtol    accepted rejected  rej.frac   nfev   pos.err   | scipy RK45: steps nfev  (nfev-2)/6-steps")
for rtol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    atol = rtol*1e-3
    t, y, st = integrate(kepler, 0.0, Y0, T4, rtol, atol, 1e-3, controller="I")
    s = solve_ivp(kepler, (0, T4), Y0, method="RK45", rtol=rtol, atol=atol)
    n_sp = len(s.t)-1; rej_sp = (s.nfev-2)/6 - n_sp
    print(f" {rtol:<7.0e} {st['accepted']:7d} {st['rejected']:7d}   {st['rejected']/(st['accepted']+st['rejected']):6.3f}  {st['nfev']:6d}  "
          f"{np.hypot(y[0]-Y0[0], y[1]-Y0[1]):.2e}  |  {n_sp:6d} {s.nfev:6d}   {rej_sp:6.1f}")
# where do rejections happen? radius at rejected attempts, for rtol=1e-6
for rtol in [1e-6, 1e-9]:
    t, y, st = integrate(kepler, 0.0, Y0, T4, rtol, rtol*1e-3, 1e-3, controller="I")
    r = np.hypot(st['rej_y'][:,0], st['rej_y'][:,1])
    print(f"rtol={rtol:g}: rejected attempts started at r in [{r.min():.3f}, {r.max():.3f}]; fraction with r<0.5: {np.mean(r<0.5):.2f}; with r>1.5: {np.mean(r>1.5):.2f}")
