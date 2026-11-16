"""Ex 1.6.F: bisection vs regula falsi vs Illinois.
Criterion: number of iterations until |x_k - x*| <= 1e-10 (x* from brentq)."""
import numpy as np
from scipy.optimize import brentq
def bisect(f, a, b, xs, tol, maxit=1000):
    fa = f(a)
    for k in range(1, maxit+1):
        c = 0.5*(a+b); fc = f(c)
        if abs(c - xs) <= tol: return c, k
        if fa*fc < 0: b = c
        else: a, fa = c, fc
def falsi(f, a, b, xs, tol, illinois=False, maxit=100000):
    fa, fb = f(a), f(b); side = 0
    for k in range(1, maxit+1):
        c = (a*fb - b*fa)/(fb - fa); fc = f(c)
        if abs(c - xs) <= tol: return c, k
        if fa*fc < 0:
            b, fb = c, fc
            if illinois and side == -1: fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if illinois and side == +1: fb *= 0.5
            side = +1
    return c, maxit
tol = 1e-10
cases = [("tan(pi-x)-x", lambda x: np.tan(np.pi-x)-x, 1.6, 3.0),
         ("x^10-1",      lambda x: x**10-1,          0.0, 1.3)]
for name, f, a, b in cases:
    xs = brentq(f, a, b, xtol=1e-15)
    print(f"{name}: root={xs:.12f}, f(a)={f(a):.4f}, f(b)={f(b):.4f}")
    for lab, fn in [("bisection", lambda: bisect(f, a, b, xs, tol)),
                    ("falsi", lambda: falsi(f, a, b, xs, tol)),
                    ("Illinois", lambda: falsi(f, a, b, xs, tol, True))]:
        r, k = fn(); print(f"  {lab:9s} {k:5d} iters")
for name, f, a, b in cases:
    fa, fb = f(a), f(b); fixed = []
    for k in range(8):
        c = (a*fb-b*fa)/(fb-fa); fc = f(c)
        if fa*fc < 0: b, fb = c, fc; fixed.append('a')
        else: a, fa = c, fc; fixed.append('b')
    print(f"{name} falsi: endpoint that stays fixed in first 8 iters: {fixed}; a={a:.6f}, b={b:.6f}")
