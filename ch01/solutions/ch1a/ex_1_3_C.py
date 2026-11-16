# 연습문제 1.3.C: Hilbert 행렬 조건수와 H_n x = 1 의 실제 정확도
import numpy as np
from scipy.linalg import hilbert
from fractions import Fraction as F
import sympy as sp
eps = np.finfo(float).eps
print(" n   kappa_2      log10k  pred.digits(16-log10k)  relerr(solve)  correct digits  ||x||_inf(exact)")
for n in [2, 4, 6, 8, 10]:
    H = hilbert(n)
    k = np.linalg.cond(H, 2)
    Hs = sp.Matrix(n, n, lambda i, j: sp.Rational(1, i + j + 1))
    xe = Hs.LUsolve(sp.ones(n, 1))
    xe = np.array([float(v) for v in xe])
    x = np.linalg.solve(H, np.ones(n))
    rel = np.linalg.norm(x - xe) / np.linalg.norm(xe)
    print(f"{n:2d}  {k:10.3e}  {np.log10(k):6.2f}   {-np.log10(eps)-np.log10(k):6.2f}             {rel:9.2e}     {-np.log10(rel):6.2f}        {np.abs(xe).max():.4g}")
    if n == 4: print("     exact x (n=4) =", [str(v) for v in Hs.LUsolve(sp.ones(n,1))])
print("eps =", eps, " -log10 eps =", -np.log10(eps))
