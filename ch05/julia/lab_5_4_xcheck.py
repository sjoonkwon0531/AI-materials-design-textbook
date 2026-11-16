"""Cross-check for the deterministic part of lab_5_4.jl: matrix exponential column, companion-matrix
cubic roots (np.roots method), fixed-step RK4 (h=1e-3) for the Schloegl ODE, stiffness ratios."""
import numpy as np
from scipy.linalg import expm
k = [1001.5, 0.5, 1.0, 0.0]; A = np.array([[-k[0], 0, 0], [k[0], -k[1], k[2]], [0, k[1], -k[2]]])
print("exact x3(t):", " ".join(f"{(expm(A*t)@[1,0,0])[2]:.4f}" for t in [0.01, 0.5, 1, 2, 4, 8]),
      "(chapter 0.0045 0.1756 0.2588 0.3167 0.3325 0.3333)")
for kf in (10., 100., 1000.):
    AB = np.array([[-kf, kf, 0], [kf, -0.5-kf, 1.0], [0, 0.5, -1.0]]); ev = np.abs(np.linalg.eigvals(AB))
    nz = np.sort(ev[ev > 1e-9]); print(f"k_fast={kf}: S={nz[-1]/nz[0]:.1f}", end="  ")
print("(chapter 16.4 160.4 1600.4)")
sc = 4.0; c = np.array([-1e-4*sc*sc/6, 3e-7*sc*1e5/2, -3.5, 1e-3*2e5/sc]); a = c/c[0]
C = np.array([[-a[1], -a[2], -a[3]], [1, 0, 0], [0, 1, 0.]]); roots = np.sort(np.linalg.eigvals(C).real)
print("roots:", np.round(roots, 1), "(chapter [ 21.4  61.9 141.7])")
f = lambda x: 3e-7*sc*1e5/2*x*x - 1e-4*sc*sc/6*x**3 + 2e2/sc - 3.5*x
def rk4(x, T=50.0, h=1e-3):
    for _ in range(int(round(T/h))):
        k1 = f(x); k2 = f(x+0.5*h*k1); k3 = f(x+0.5*h*k2); k4 = f(x+h*k3); x += h/6*(k1+2*k2+2*k3+k4)
    return x
for x0 in (roots[1]-2, roots[1]+2): print(f"ODE from x0={x0:.1f}: x(50)={rk4(x0):.1f}", end="  ")
print("(chapter 59.9 -> 21.4, 63.9 -> 141.7)")
