# Ex 2.7 / 2.2.P(a): RK4 stability region |g(z)|<=1, g = 1+z+z^2/2+z^3/6+z^4/24
import numpy as np, sympy as sp
from scipy.optimize import brentq
g = lambda z: 1+z+z**2/2+z**3/6+z**4/24
# real-axis interval
xr = brentq(lambda x: abs(g(x))-1, -3.5, -2.0)
print(f"real-axis stability interval: [{xr:.6f}, 0]")
# imaginary axis: |g(iy)|^2 = 1 - y^6/72 + y^8/576
y = sp.symbols('y', real=True)
mod2 = sp.expand(sp.simplify((g(sp.I*y)*sp.conjugate(g(sp.I*y)))))
print("|g(iy)|^2 =", mod2)
yi = [s for s in sp.solve(sp.Eq(mod2, 1), y) if s.is_positive][0]
print(f"imag-axis interval: |y| <= {yi} = {float(yi):.6f}")
# boundedness: scan region on grid
X, Y = np.meshgrid(np.linspace(-5, 2, 1401), np.linspace(-4, 4, 1601))
Z = X + 1j*Y
S = np.abs(g(Z)) <= 1
print(f"stable grid points with Re z<-3: {np.sum(S & (X < -3.0))}, with |Im z|>3: {np.sum(S & (np.abs(Y) > 3.0))}")
print(f"max |z| over stable grid points: {np.abs(Z[S]).max():.4f}")
for r in [5, 10, 100]:
    th = np.linspace(0, 2*np.pi, 3601)
    print(f"min |g| on circle |z|={r}: {np.abs(g(r*np.exp(1j*th))).min():.3e}")
# example: z = -3 (real, outside), z=-10
print(f"g(-3) = {g(-3):.4f}, g(-10) = {g(-10):.1f}")
# plot (optional)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = "Arial Narrow" if any("Arial Narrow" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 14})
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.contourf(X, Y, S.astype(float), levels=[0.5, 1.5], colors=["#9ecae1"])
    ax.contour(X, Y, np.abs(g(Z)), levels=[1], colors="k")
    ee = np.abs(1+Z) <= 1; ax.contour(X, Y, ee.astype(float), levels=[0.5], colors="gray", linestyles="--")
    ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
    ax.set_xlabel("Re z"); ax.set_ylabel("Im z"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig("ex2_7_rk4_stability.png", dpi=150)
    print("saved ex2_7_rk4_stability.png")
except Exception as e:
    print("plot skipped:", e)
# crude analytic bound: |g| >= r^4/24 - (1 + r + r^2/2 + r^3/6) > 1  for r > R
R = brentq(lambda r: r**4/24 - (2 + r + r**2/2 + r**3/6), 3, 20)
print(f"triangle-inequality bound: region contained in |z| < {R:.4f}")
