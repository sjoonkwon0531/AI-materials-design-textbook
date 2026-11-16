"""Ex 5.2.3: NPT volume moves for an ideal gas (U = 0) where pi(V) ~ V^N exp(-beta P V) is known exactly.
N = 10, beta P = 1  ->  <V> = (N+1)/(beta P) = 11, Var V = (N+1)/(beta P)^2 = 11.
Three proposal schemes with their Jacobian coefficients, plus the 'wrong' ln V scheme without the term."""
import numpy as np
from numba import njit
N = 10; bP = 1.0

@njit(cache=True)
def run(scheme, coef, nstep, seed, step):
    np.random.seed(seed)
    V = 11.0; s = 0.0; s2 = 0.0; Vmin = 1e300
    for k in range(nstep):
        if scheme == 0:          # ln V uniform
            Vn = V * np.exp(step * (2 * np.random.random() - 1))
        elif scheme == 1:        # V uniform
            Vn = V + step * (2 * np.random.random() - 1)
        else:                    # L uniform (cube)
            L = V ** (1 / 3); Ln = L + step * (2 * np.random.random() - 1)
            Vn = Ln ** 3 if Ln > 0 else -1.0
        if Vn > 0:
            arg = -bP * (Vn - V) + coef * np.log(Vn / V)
            if arg >= 0 or np.random.random() < np.exp(arg):
                V = Vn
        s += V; s2 += V * V
        if V < Vmin: Vmin = V
    m = s / nstep
    return m, s2 / nstep - m * m, Vmin

cases = [("ln V uniform, coef N+1", 0, N + 1, 0.3), ("V uniform, coef N", 1, N, 3.0),
         ("L uniform, coef N+2/3", 2, N + 2 / 3, 0.3), ("V uniform, coef N+1 (wrong)", 1, N + 1, 3.0),
         ("ln V uniform, coef 0 (wrong)", 0, 0.0, 0.3)]
print(f"exact: <V> = {(N+1)/bP:.3f}, Var V = {(N+1)/bP**2:.3f}")
for name, sc, cf, st in cases:
    m, v, vmin = run(sc, cf, 4_000_000, 1, st)
    print(f"{name:30s}: <V> = {m:8.3f}   Var V = {v:8.3f}   min V visited = {vmin:.2e}")
# size of the residual 1/V term for the wrong LJ run of Lab 5.1 (N = 256, rho = 0.6503, T* = 2)
Vw = 256 / 0.6503
print(f"LJ wrong run: V = {Vw:.1f}, kT/V = {2.0/Vw:.5f}, ideal-gas pressure rho kT = {0.6503*2.0:.4f}")
