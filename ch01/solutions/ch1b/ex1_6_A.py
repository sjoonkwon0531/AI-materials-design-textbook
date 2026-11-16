"""Ex 1.6.A: x^3-2x-5=0 by bisection, Newton, secant."""
import numpy as np
from scipy.optimize import brentq
f  = lambda x: x**3 - 2*x - 5
df = lambda x: 3*x**2 - 2
xs = brentq(f, 2, 3, xtol=1e-15, rtol=4*np.finfo(float).eps)
print(f"root = {xs:.16f}")

# bisection: stop when interval can no longer shrink (machine precision)
a, b, k = 0.0, 3.0, 0
while True:
    c = 0.5*(a+b)
    if c == a or c == b: break
    if f(a)*f(c) < 0: b = c
    else: a = c
    k += 1
print(f"bisection: {k} iters, x = {0.5*(a+b):.16f}, err = {abs(0.5*(a+b)-xs):.1e}")
print(f"  theory log2(3/eps(2)) = {np.log2(3/np.spacing(2.0)):.1f}")

x, k = 2.0, 0
print("Newton errors:")
while True:
    xn = x - f(x)/df(x); k += 1
    print(f"  k={k}: x={xn:.16f}, |e|={abs(xn-xs):.2e}")
    if xn == x or k > 50: break
    x = xn
x0, x1, k = 2.0, 3.0, 0
print("Secant errors:")
while True:
    if f(x1) == f(x0): break
    x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0)); k += 1
    print(f"  k={k}: x={x2:.16f}, |e|={abs(x2-xs):.2e}")
    if x2 == x1 or k > 60: break
    x0, x1 = x1, x2
