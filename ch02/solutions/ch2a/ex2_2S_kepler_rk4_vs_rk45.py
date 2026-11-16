# Ex 2.2.S: fixed-step RK4 vs adaptive RK45 on the e=0.8 Kepler orbit (4 periods) at equal accuracy
import numpy as np
from scipy.integrate import solve_ivp
from kepler import kepler, Y0, T4, energy
def rk4_fixed(N):
    h = T4/N; Y = Y0.copy(); t = 0.0
    for _ in range(N):
        k1=kepler(t,Y); k2=kepler(t+h/2,Y+h/2*k1); k3=kepler(t+h/2,Y+h/2*k2); k4=kepler(t+h,Y+h*k3)
        Y = Y + h/6*(k1+2*k2+2*k3+k4); t += h
    return Y
perr = lambda Y: np.hypot(Y[0]-Y0[0], Y[1]-Y0[1])     # position error after 4 periods (exact: back at Y0)
for rtol, atol in [(1e-6, 1e-9), (1e-8, 1e-11)]:
    s = solve_ivp(kepler, (0, T4), Y0, method="RK45", rtol=rtol, atol=atol)
    e45 = perr(s.y[:, -1])
    # smallest N (to ~1%) with RK4 error <= e45: geometric scan, then bisection
    N = 100
    while perr(rk4_fixed(N)) > e45: N *= 2
    lo, hi = N//2, N
    while hi - lo > max(1, lo//100):
        mid = (lo+hi)//2
        if perr(rk4_fixed(mid)) <= e45: hi = mid
        else: lo = mid
    e4 = perr(rk4_fixed(hi))
    print(f"rtol={rtol:g}: RK45 steps={len(s.t)-1}, nfev={s.nfev}, pos.err={e45:.2e}, |dE|={abs(energy(s.y[:,-1])-energy(Y0)):.1e}, "
          f"dt range [{np.diff(s.t)[1:-1].min():.4f}, {np.diff(s.t)[1:-1].max():.4f}]")
    print(f"          RK4 fixed: N={hi} (dt={T4/hi:.5f}), nfev={4*hi}, pos.err={e4:.2e}  -> nfev ratio RK4/RK45 = {4*hi/s.nfev:.1f}")
