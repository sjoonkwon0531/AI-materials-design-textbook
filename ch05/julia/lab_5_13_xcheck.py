"""Cross-check for Part 0 of lab_5_13.jl: golden-section minimisation (same bracket update as the Julia
fminbound) of N*Gamma(x) for continuous Gaussian chains; compare with leibler_check.py (scipy bounded)."""
import math
g = lambda f, x: 2*(f*x+math.exp(-f*x)-1)/x**2
def NG(x, f=0.5):
    saa = g(f, x); sbb = g(1-f, x); sab = 0.5*(g(1, x)-saa-sbb); return (saa+sbb+2*sab)/(saa*sbb-sab**2)
def fminbound(f, a, b, tol=1e-9):
    gr = (math.sqrt(5)-1)/2; c = b-gr*(b-a); d = a+gr*(b-a); fc = f(c); fd = f(d)
    while b-a > tol:
        if fc < fd: b, d, fd = d, c, fc; c = b-gr*(b-a); fc = f(c)
        else: a, c, fc = c, d, fd; d = a+gr*(b-a); fd = f(d)
    return (a+b)/2
x = fminbound(NG, 0.5, 10.0)
print(f"x* = (q* Rg)^2 = {x:.4f},  q* Rg = {math.sqrt(x):.4f},  (chi N)_s = {NG(x)/2:.4f}")
print("leibler_check.py: x* = (q* Rg)^2 = 3.7852,  q* Rg = 1.9456,  (chi N)_s = 10.4949")
