# Cross-check of lab_1_8.jl: forward-mode dual numbers (ForwardDiff analogue) in pure Python
import math
class D:
    def __init__(s, a, b): s.a, s.b = a, b
    def __mul__(s, o): return D(s.a*o.a, s.a*o.b + s.b*o.a)
def dsin(x): return D(math.sin(x.a), math.cos(x.a)*x.b)
def dsq(x): return D(x.a**2, 2*x.a**1*x.b)          # ForwardDiff: x^2 -> 2x^(2-1) * dx
f = lambda x: x**2*math.sin(x)
x0 = 1.0; exact = 2*1.0*math.sin(1.0) + 1.0**2*math.cos(1.0)
ad = (dsq(D(x0, 1.0)) * dsin(D(x0, 1.0))).b
res = [("Analytical (by hand)", 2*x0*math.sin(x0) + x0**2*math.cos(x0)),
       ("Symbolic (Symbolics)", 2*x0*math.sin(x0) + x0**2*math.cos(x0)),
       ("Numerical (h=1e-5)", (f(x0+1e-5)-f(x0-1e-5))/(2*1e-5)),
       ("Numerical (h=1e-10)", (f(x0+1e-10)-f(x0-1e-10))/(2*1e-10)),
       ("Automatic (ForwardDiff)", ad)]
print(f"{'Method':25s} | {'Value':>20s} | {'|Error|':>10s}"); print("-"*65)
for n, v in res: print(f"{n:25s} | {v:20.16f} | {abs(v-exact):10.2e}")
