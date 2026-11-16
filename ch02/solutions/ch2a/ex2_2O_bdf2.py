# Ex 2.2.O: hand-written BDF2 on the Lab 2.3 stiff network; order-2 convergence
import numpy as np
from scipy.linalg import expm, lu_factor, lu_solve
from stiffnet import A, X0, T
I = np.eye(3)
lam, V = np.linalg.eig(A); c = np.linalg.solve(V, X0)
exact = lambda t: (V*np.exp(lam*t))@c                       # eigendecomposition baseline (Box 2.1)
def bdf2(h, start="be"):
    n = int(round(T/h))
    lu1 = lu_factor(I - h*A); lu2 = lu_factor(3*I - 2*h*A)  # BDF2: 3X_{n+1}-4X_n+X_{n-1} = 2h A X_{n+1}
    Xm = X0.copy(); X = lu_solve(lu1, X0) if start == "be" else exact(h)
    emax = np.abs(X - exact(h)).max()
    for j in range(2, n+1):
        X, Xm = lu_solve(lu2, 4*X - Xm), X
        emax = max(emax, np.abs(X - exact(j*h)).max())
    return np.abs(X - exact(T)).max(), emax
def be(h):
    n = int(round(T/h)); lu = lu_factor(I - h*A); X = X0.copy(); emax = 0
    for j in range(1, n+1):
        X = lu_solve(lu, X); emax = max(emax, np.abs(X - exact(j*h)).max())
    return np.abs(X - exact(T)).max(), emax
def table(name, fn, hs, col):
    print(name); prev = None
    for h in hs:
        e = fn(h); ec = e[col]
        s = "" if prev is None else f"  ratio={prev/ec:6.2f}  p={np.log2(prev/ec):.2f}"
        print(f"  dt={h:<9} err(t=8)={e[0]:.3e}  max_t err={e[1]:.3e}{s}"); prev = ec
print("Regime 1: dt >> 1/|lam_max| (fast transient not resolved), error measured at t=8")
table("BDF2 (BE start)", bdf2, [0.1, 0.05, 0.025, 0.0125, 0.00625], 0)
table("BDF1 = implicit Euler", be, [0.1, 0.05, 0.025, 0.0125, 0.00625], 0)
print("Regime 2: dt << 1/|lam_max| (transient resolved), max error over the whole trajectory")
table("BDF2 (BE start)", bdf2, [4e-4, 2e-4, 1e-4, 5e-5], 1)
table("BDF2 (exact start)", lambda h: bdf2(h, "ex"), [4e-4, 2e-4, 1e-4, 5e-5], 1)
print("BDF2 dt=0.5 (250x the explicit-Euler limit): err(t=8) = %.3e" % bdf2(0.5)[0])
