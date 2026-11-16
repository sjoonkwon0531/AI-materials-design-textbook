# Ex 2.2.L: u'' = u^2, u(0)=0, u(1)=1 by shooting; residual scan, Newton via variational equation
import numpy as np
from scipy.integrate import solve_bvp
N = 2000; dx = 1.0/N
def F(Y):            # Y = [u, u', w, w'],  w = du/ds:  w'' = 2 u w,  w(0)=0, w'(0)=1
    u, up, w, wp = Y
    return np.array([up, u*u, wp, 2*u*w])
def shoot(s):
    Y = np.array([0.0, s, 0.0, 1.0]); umin = 0.0
    for _ in range(N):
        k1=F(Y); k2=F(Y+dx/2*k1); k3=F(Y+dx/2*k2); k4=F(Y+dx*k3); Y = Y+dx/6*(k1+2*k2+2*k3+k4)
        if not np.all(np.isfinite(Y)) or abs(Y[0]) > 1e8: return np.inf, np.nan, umin
        umin = min(umin, Y[0])
    return Y[0]-1.0, Y[2], umin           # g(s), g'(s)=w(1), min u
# (1) residual is not affine
for s in [-40, -30, -10, 0, 1, 2, 5]:
    gv = shoot(s)[0]; print(f"g({s:>4}) = {gv:+.6f}")
G = [shoot(s)[0] for s in (0.0, 1.0, 2.0)]
print(f"finite-difference slopes on [0,1],[1,2]: {G[1]-G[0]:.4f}, {G[2]-G[1]:.4f}  (linear BVP would give equal slopes)")
# (2) Newton
def newton(s, tol=1e-12, maxit=30):
    hist = []
    for k in range(maxit):
        gv, dg, umin = shoot(s)
        if not np.isfinite(gv): hist.append((s, np.inf)); return s, hist, False, umin
        hist.append((s, gv))
        if abs(gv) < tol: return s, hist, True, umin
        s = s - gv/dg
    return s, hist, False, umin
for s0 in [0.0, 1.0, 5.0, -10.0, -20.0, -30.0, -50.0, -100.0, -300.0]:
    s, hist, ok, umin = newton(s0)
    print(f"s0={s0:>7}: converged={ok}, iters={len(hist)-1}, s*={s:.10f}, min u={umin:.6f}, |g| history: " +
          ", ".join(f"{abs(h[1]):.1e}" for h in hist[:8]))
# (3) cross-check with a collocation BVP solver from two initial guesses
x = np.linspace(0, 1, 101)
for guess, lab in [(np.vstack([x, np.ones_like(x)]), "u0=x"), (np.vstack([-60*x*(1-x)+x, -60+120*x+1]), "u0=-60x(1-x)+x")]:
    sol = solve_bvp(lambda x, Y: np.vstack([Y[1], Y[0]**2]), lambda a, b: np.array([a[0], b[0]-1]), x, guess, tol=1e-8, max_nodes=100000)
    print(f"solve_bvp from {lab}: status={sol.status}, u'(0)={sol.sol(0)[1]:.8f}, min u={sol.sol(np.linspace(0,1,2001))[0].min():.6f}")
