# Ex 2.2.M: FDM (tridiagonal) vs shooting for the Lab 2.5 rod
import numpy as np, time
from scipy.linalg import solve_banded
kap, Ta, L, T0, TL = 0.01, 20.0, 10.0, 40.0, 200.0
m = np.sqrt(kap)
c2 = (TL - Ta - (T0-Ta)*np.exp(-m*L)) / (2*np.sinh(m*L)) ; c1 = (T0-Ta) - c2   # T = c1 e^{-mx} + c2 e^{mx} + Ta
Tex = lambda x: c1*np.exp(-m*x) + c2*np.exp(m*x) + Ta
print(f"analytic: T = {c2:.4f} e^(0.1x) + ({c1:.4f}) e^(-0.1x) + 20")
def fdm(n):                    # n intervals, n-1 unknowns: (T_{i-1} - (2+kap h^2) T_i + T_{i+1}) = -kap h^2 Ta
    h = L/n; x = np.linspace(0, L, n+1); N = n-1
    ab = np.zeros((3, N)); ab[0, 1:] = 1.0; ab[1, :] = -(2 + kap*h*h); ab[2, :-1] = 1.0
    F = np.full(N, -kap*h*h*Ta); F[0] -= T0; F[-1] -= TL
    T = np.empty(n+1); T[0], T[-1] = T0, TL; T[1:-1] = solve_banded((1, 1), ab, F)
    return x, T
prev = None
for n in [10, 100, 1000, 10000]:
    t0 = time.perf_counter(); x, T = fdm(n); dt_ = time.perf_counter()-t0
    e = np.abs(T - Tex(x)).max()
    print(f"FDM h={L/n:<7g} unknowns={n-1:6d}  max err={e:.3e}" + ("" if prev is None else f"  ratio={prev/e:.1f}") + f"  solve {dt_*1e3:.2f} ms"); prev = e
# Richardson extrapolation of h=0.01 and h=0.005 (O(h^4))
x1, T1 = fdm(1000); x2, T2 = fdm(2000); Tr = (4*T2[::2] - T1)/3
print(f"Richardson (h=0.01,0.005): max err={np.abs(Tr-Tex(x1)).max():.3e}")
# shooting (RK4 dx=0.01, secant) for comparison
rhs = lambda Y: np.array([Y[1], kap*(Y[0]-Ta)])
def shoot(s, dx=0.01, keep=False):
    Y = np.array([T0, s]); out = [Y[0]]
    for _ in range(int(round(L/dx))):
        k1=rhs(Y); k2=rhs(Y+dx/2*k1); k3=rhs(Y+dx/2*k2); k4=rhs(Y+dx*k3); Y = Y+dx/6*(k1+2*k2+2*k3+k4); out.append(Y[0])
    return np.array(out) if keep else Y[0]
t0 = time.perf_counter()
g0, g1 = shoot(0.0)-TL, shoot(500.0)-TL; s = 500.0 - g1*500.0/(g1-g0); Ts = shoot(s, keep=True)
dts = time.perf_counter()-t0
print(f"shooting RK4 dx=0.01 + secant: max err={np.abs(Ts - Tex(np.linspace(0, L, 1001))).max():.3e}, 3 shots = {3*1000*4} f-evals, {dts*1e3:.1f} ms (python loop)")
