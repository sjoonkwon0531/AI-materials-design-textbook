# Cross-check of lab_1_6.jl pointwise-loop logic (Numba mirrors Julia's scalar loop)
import numpy as np
from numba import njit
roots = np.array([1.0, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)])
N, span, max_iter, tol = 1000, 2.0, 50, 1e-6
@njit
def newton_point(z, roots):
    for k in range(1, max_iter + 1):
        df = 3.0 * z**2
        step = (z**3 - 1.0) / df if abs(df) > 1e-12 else 0j
        z = z - step
        for ri in range(3):
            if abs(z - roots[ri]) < tol:
                return ri + 1, k
    return 0, 0
@njit
def run(x, roots):
    basin = np.zeros((N, N), np.int64); itc = np.zeros((N, N), np.int64)
    for j in range(N):
        for i in range(N):
            b, k = newton_point(complex(x[j], x[i]), roots)
            basin[i, j] = b; itc[i, j] = k
    return basin, itc
x = np.linspace(-span, span, N)
basin, itc = run(x, roots)
conv = basin > 0
print(f"converged fraction = {conv.mean():.4f}")
print(f"mean iterations    = {itc[conv].mean():.3f}")
print("basin fractions    =", [round(float(np.mean(basin == i)), 4) for i in (1, 2, 3)])
print(f"max iterations     = {itc.max()}")
