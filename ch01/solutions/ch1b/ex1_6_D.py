"""Ex 1.6.D: N2+3H2 <-> 2NH3 equilibrium by 3-variable Newton.
Illustrative (assumed) K_P = 1.5e-5 bar^-2, P = 200 bar, feed 1 mol N2 + 3 mol H2."""
import numpy as np, time
from scipy.optimize import brentq
KP, P = 1.5e-5, 200.0
lnKP2 = np.log(KP*P**2)
def F(n):
    n1, n2, n3 = n; nt = n.sum()
    return np.array([2*n1 + n3 - 2.0,                 # N balance
                     2*n2 + 3*n3 - 6.0,               # H balance
                     2*np.log(n3) - np.log(n1) - 3*np.log(n2) + 2*np.log(nt) - lnKP2])
def J_an(n):
    n1, n2, n3 = n; nt = n.sum()
    return np.array([[2, 0, 1], [0, 2, 3],
                     [-1/n1 + 2/nt, -3/n2 + 2/nt, 2/n3 + 2/nt]], float)
def J_fd(n, h=1e-7):
    f0 = F(n); J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = h*max(1, abs(n[j]))
        J[:, j] = (F(n+e) - f0)/e[j]
    return J
def newton(Jf, n0, tol=1e-14, maxit=50, verbose=False):
    n = n0.copy(); hist = []
    for k in range(maxit):
        r = np.linalg.norm(F(n)); hist.append(r)
        if r < tol: break
        dn = np.linalg.solve(Jf(n), -F(n))
        lam = 1.0                       # keep moles positive (damping)
        while np.any(n + lam*dn <= 0): lam *= 0.5
        n = n + lam*dn
    return n, k, hist
n0 = np.array([0.9, 2.7, 0.2])          # xi = 0.1 guess
for name, Jf in [("analytic", J_an), ("finite-diff", J_fd)]:
    n, k, hist = newton(Jf, n0)
    t = time.perf_counter()
    for _ in range(2000): newton(Jf, n0)
    dt = (time.perf_counter()-t)/2000
    print(f"{name}: iters={k}, n={n}, y={n/n.sum()}, |F|={hist[-1]:.1e}, time/solve={dt*1e6:.1f} us")
    print("   residual history:", ["%.1e" % h for h in hist])
# reference by 1D extent
g = lambda xi: 2*np.log(2*xi) - np.log(1-xi) - 3*np.log(3-3*xi) + 2*np.log(4-2*xi) - lnKP2
xi = brentq(g, 1e-12, 1-1e-12, xtol=1e-15)
nref = np.array([1-xi, 3-3*xi, 2*xi])
print(f"reference xi = {xi:.15f}, y_NH3 = {2*xi/(4-2*xi):.6f}")
for name, Jf in [("analytic", J_an), ("finite-diff", J_fd)]:
    n, k, _ = newton(Jf, n0); print(f"  {name}: max|n - nref| = {np.abs(n-nref).max():.1e}")
# Jacobian accuracy at the initial point
print("max |J_fd - J_an| at n0 =", np.abs(J_fd(n0)-J_an(n0)).max())
for h in [1e-4, 1e-7, 1e-10]:
    print(f"  h={h:g}: max|J_fd-J_an| = {np.abs(J_fd(n0,h)-J_an(n0)).max():.1e}")
