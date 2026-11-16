"""Leibler (1980) mean-field spinodal of a symmetric diblock from continuous Gaussian (Debye) form factors."""
import numpy as np
from scipy.optimize import minimize_scalar
g = lambda f, x: 2 * (f * x + np.exp(-f * x) - 1) / x ** 2        # x = q^2 Rg^2
def Gam(x, f=0.5):                                                   # N * Gamma(q)
    saa = g(f, x); sbb = g(1 - f, x); sab = 0.5 * (g(1, x) - saa - sbb)
    return (saa + sbb + 2 * sab) / (saa * sbb - sab ** 2)
r = minimize_scalar(Gam, bounds=(0.5, 10), method="bounded")
print(f"x* = (q* Rg)^2 = {r.x:.4f},  q* Rg = {np.sqrt(r.x):.4f},  (chi N)_s = {Gam(r.x)/2:.4f}")
