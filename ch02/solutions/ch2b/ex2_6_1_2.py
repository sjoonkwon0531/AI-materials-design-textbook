# Ex 2.6.1 (double well) and 2.6.2 (fastest mode vs eps; Lab 2.14 with eps=0.05)
import numpy as np, sympy as sp
from ch_common import ch_run, peak_k, length_scale
c = sp.Symbol('c'); f = (c**2-1)**2/4
f1, f2 = sp.diff(f, c), sp.diff(f, c, 2)
print("f' =", sp.factor(f1), "  f'' =", f2, "  f''(0) =", f2.subs(c, 0), "  f''(+-1) =", f2.subs(c, 1))
print("spinodal f''=0 at c = +-1/sqrt(3) =", [float(s) for s in sp.solve(f2, c)])

print("\n2.6.2 theory: k* = 1/(eps sqrt2), lambda* = 2 pi sqrt2 eps, sigma_max = 1/(4 eps^2)")
L = 4*np.pi
for eps in [0.1, 0.05]:
    ks = 1/(eps*np.sqrt(2))
    print(f"  eps={eps}: k*={ks:.3f}  lambda*={2*np.pi/ks:.4f}  wavelengths across L=4pi: {ks*L/(2*np.pi):.2f}  sigma_max={1/(4*eps**2):.0f}")

# early-time pattern: run until sigma_max * t = 4 (still linear-ish), small dt for accuracy
for eps, N in [(0.1, 256), (0.05, 256), (0.05, 512)]:
    t_lin = 4*4*eps**2
    cfin = ch_run(N=N, L=L, eps=eps, dt=t_lin/400, S=2.0, T=t_lin, seed=1)
    print(f"eps={eps:4.2f} N={N} dx={L/N:.4f}: t={t_lin:.3f}  S(k) peak k={peak_k(cfin):.2f}  (theory {1/(eps*np.sqrt(2)):.2f}),  max|c|={np.abs(cfin).max():.3f}")
# the Lab's own snapshot time t=1 (dt=0.05)
for eps in [0.1, 0.05]:
    cfin = ch_run(eps=eps, T=1.0, seed=1)
    print(f"Lab settings, t=1, eps={eps}: L(t)=2pi/<k> = {length_scale(cfin):.3f},  S(k) peak k = {peak_k(cfin):.2f}")

# discrete-map growth of the stabilized scheme around c=0: g(k) = 1 + dt (k^2 - eps^2 k^4)/(1 + dt S k^2 + dt eps^2 k^4)
k = np.linspace(0.01, 30, 300001)
for eps in [0.1, 0.05]:
    for dt in [0.05, 1e-4]:
        g = 1 + dt*(k**2 - eps**2*k**4)/(1 + dt*2.0*k**2 + dt*eps**2*k**4)
        print(f"scheme (S=2, dt={dt:g}) eps={eps}: fastest-growing k of the discrete map = {k[g.argmax()]:.2f} (continuous k*={1/(eps*np.sqrt(2)):.2f}), g_max={g.max():.4f}")
