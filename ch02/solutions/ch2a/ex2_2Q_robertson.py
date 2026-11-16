# Ex 2.2.Q: Robertson problem, BDF vs RK45 (and Radau, LSODA) function evaluations
import numpy as np, time, sys
# default: t in [0,40] (~5 s). `python3 ex2_2Q_robertson.py --long` adds t in [0,1000] (RK45 takes ~5 min).
from scipy.integrate import solve_ivp
def rob(t, y):
    y1, y2, y3 = y
    return [-0.04*y1 + 1e4*y2*y3, 0.04*y1 - 1e4*y2*y3 - 3e7*y2**2, 3e7*y2**2]
def jac(t, y):
    y1, y2, y3 = y
    return [[-0.04, 1e4*y3, 1e4*y2], [0.04, -1e4*y3-6e7*y2, -1e4*y2], [0, 6e7*y2, 0]]
y0 = [1.0, 0.0, 0.0]
# Jacobian eigenvalues near t ~ 40 indicate stiffness
ref = solve_ivp(rob, (0, 1e5), y0, method="Radau", jac=jac, rtol=1e-12, atol=1e-14, dense_output=True)
for tt in [0.0, 1.0, 40.0]:
    ev = np.linalg.eigvals(np.array(jac(0, ref.sol(tt)))); print(f"t={tt:>5}: Jacobian eigenvalues = {np.sort(ev.real)}")
for T in ([40.0, 1e3] if "--long" in sys.argv else [40.0]):
    print(f"--- t in [0, {T:g}], rtol=1e-6, atol=1e-10 ---")
    yref = ref.sol(T)
    for m in ["BDF", "Radau", "LSODA", "RK45"]:
        t0 = time.time()
        kw = dict(jac=jac) if m in ("BDF", "Radau", "LSODA") else {}
        s = solve_ivp(rob, (0, T), y0, method=m, rtol=1e-6, atol=1e-10, **kw)
        relerr = np.abs((s.y[:, -1]-yref)/np.maximum(np.abs(yref), 1e-20))
        print(f"  {m:6s} success={s.success} steps={len(s.t)-1:7d} nfev={s.nfev:8d} njev={s.njev:4d} nlu={s.nlu:5d} "
              f"wall={time.time()-t0:6.2f}s  max rel.err={relerr.max():.1e}  y({T:g})={s.y[:, -1]}")
    print(f"  reference y({T:g}) = {yref}, sum = {yref.sum():.15f}")
# effect of a too-loose absolute tolerance on the tiny intermediate y2 ~ 1e-5
yref = ref.sol(40.0)
for atol in [1e-6, 1e-10]:
    s = solve_ivp(rob, (0, 40), y0, method="BDF", jac=jac, rtol=1e-6, atol=atol)
    print(f"BDF atol={atol:g}: y2(40)={s.y[1,-1]:.4e} (ref {yref[1]:.4e}), rel.err y2={abs(s.y[1,-1]-yref[1])/yref[1]:.1e}, nfev={s.nfev}")
