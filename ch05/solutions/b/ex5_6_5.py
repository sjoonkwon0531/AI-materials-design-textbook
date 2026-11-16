"""Ex 5.6.5: Cauchy-Born for a 1D LJ chain with 1st+2nd neighbour interactions; finite free chain N=100."""
import numpy as np
from scipy.optimize import brentq, minimize
phi = lambda r: 4*(r**-12 - r**-6)
dphi = lambda r: 4*(-12*r**-13 + 6*r**-7)
d2phi = lambda r: 4*(156*r**-14 - 42*r**-8)
W = lambda a: phi(a) + phi(2*a)
dW = lambda a: dphi(a) + 2*dphi(2*a)
a_s = brentq(dW, 1.0, 1.2)
Wpp = d2phi(a_s) + 4*d2phi(2*a_s)
print(f"a* = {a_s:.6f} (nn only: 2^(1/6) = {2**(1/6):.6f}),  W(a*) = {W(a_s):.6f},  W''(a*) = {Wpp:.4f}  "
      f"[nn part {d2phi(a_s):.4f}, 2nd-nn part {4*d2phi(2*a_s):.4f}];  1D modulus a* W''(a*) = {a_s*Wpp:.3f}")
N = 100
def E(x):
    d1 = np.diff(x); d2 = x[2:] - x[:-2]
    return phi(d1).sum() + phi(d2).sum()
def G(x):
    g = np.zeros_like(x); d1 = np.diff(x); d2 = x[2:] - x[:-2]
    f1 = dphi(d1); f2 = dphi(d2)
    g[1:] += f1; g[:-1] -= f1; g[2:] += f2; g[:-2] -= f2
    return g
x0 = np.arange(N)*a_s
r = minimize(E, x0, jac=G, method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000))
# polish with Newton steps using the exact (banded) Hessian
def Hm(x):
    Hh = np.zeros((N, N)); d1 = np.diff(x); d2 = x[2:] - x[:-2]
    for i, k in enumerate(d2phi(d1)):
        Hh[i, i] += k; Hh[i+1, i+1] += k; Hh[i, i+1] -= k; Hh[i+1, i] -= k
    for i, k in enumerate(d2phi(d2)):
        Hh[i, i] += k; Hh[i+2, i+2] += k; Hh[i, i+2] -= k; Hh[i+2, i] -= k
    return Hh
x = r.x.copy()
for it in range(8):
    Hh = Hm(x); Hh[0, 0] += 1.0                      # pin the translation mode softly
    x = x - np.linalg.lstsq(Hm(x) + np.outer(np.ones(N), np.ones(N))/N, G(x), rcond=None)[0]
r.x = x; r.fun = E(x)
d = np.diff(r.x)
k1, k2 = d2phi(a_s), d2phi(2*a_s)
# linearised bulk equation k1(l-2+1/l) + k2(l^2-2+1/l^2) = 0 -> decaying root
lam = [z for z in np.roots([k2, k1, -2*k1-2*k2, k1, k2]) if abs(z) < 0.5]
print(f"phi''(a*)={k1:.4f}, phi''(2a*)={k2:.4f}; decaying root(s) of the linearised chain: {np.array2string(np.real_if_close(np.array(lam)), precision=5)}")
print(f"converged={r.success}, max|grad|={np.abs(G(r.x)).max():.1e}")
print("bond spacings from the end (k=1 is the outermost bond):")
for k in range(6):
    print(f"  k={k+1}: {d[k]:.6f}  (d - a*)/a* = {(d[k]-a_s)/a_s:+.3e}")
print(f"middle bond (k=50): {d[49]:.8f}, (d - a*)/a* = {(d[49]-a_s)/a_s:+.1e}")
dev = np.abs(d[:50]-a_s)
for tol in (1e-3, 1e-6, 1e-9):
    print(f"  bonds with |d-a*|/a* > {tol:g}: first {np.sum(dev/a_s > tol)} from each end")
ratio = (d[1:6]-a_s)/(d[:5]-a_s)
print("successive deviation ratios (k+1)/k:", np.array2string(ratio, precision=4))
E_cb = (N-1)*phi(a_s) + (N-2)*phi(2*a_s)
print(f"total energy: relaxed {r.fun:.6f}, uniform a* {E(x0):.6f}, N*W(a*) {N*W(a_s):.6f}; "
      f"surface energy per end (relaxed - N W) / 2 = {(r.fun - N*W(a_s))/2:.6f}; relaxation gain {(E(x0)-r.fun):.2e}")
