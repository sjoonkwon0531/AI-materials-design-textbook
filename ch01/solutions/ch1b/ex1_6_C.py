"""Ex 1.6.C: VdW CO2 molar volume at 300 K, 10 bar by Newton."""
import numpy as np
a, b, R, T, P = 3.640, 0.04267, 0.08314462618, 300.0, 10.0
f  = lambda V: (P + a/V**2)*(V - b) - R*T
df = lambda V: P - a/V**2 + 2*a*b/V**3
V0 = R*T/P
print(f"ideal V0 = {V0:.6f} L/mol")
V = V0
for k in range(1, 10):
    Vn = V - f(V)/df(V)
    print(f"  k={k}: V={Vn:.12f}, |dV|={abs(Vn-V):.2e}, |f|={abs(f(Vn)):.2e}")
    if Vn == V: break
    V = Vn
Vs = V
print(f"Z = PV/RT = {P*Vs/(R*T):.6f}")
# cubic form roots: P V^3 - (Pb+RT) V^2 + a V - ab = 0
r = np.roots([P, -(P*b + R*T), a, -a*b]); print("cubic roots:", r)
# error ratio e_{k+1}/e_k^2
V = V0; errs = []
for k in range(6):
    errs.append(abs(V-Vs)); V = V - f(V)/df(V)
print("e_k:", ["%.2e" % e for e in errs])
print("e_{k+1}/e_k^2:", ["%.3f" % (errs[i+1]/errs[i]**2) for i in range(3)])
# theoretical constant f''/(2f') at root
d2f = lambda V: 2*a/V**3 - 6*a*b/V**4
print(f"C = |f''/(2f')| at root = {abs(d2f(Vs)/(2*df(Vs))):.3f}")
# Bad start: V0 = b-ish liquid region and a start where df=0
Vcrit = np.roots([P, 0, -a, 2*a*b]); print("df=0 at V:", Vcrit)
for V0b in [0.05, 0.1, 0.3]:
    V = V0b
    for k in range(60):
        if df(V) == 0: break
        Vn = V - f(V)/df(V)
        if abs(Vn - V) < 1e-14: break
        V = Vn
    print(f"  start {V0b}: -> V={V:.6f} after {k+1} iters")
V = 0.1; hist = []
for k in range(200):
    V = V - f(V)/df(V); hist.append(V)
print("start 0.1, last iterates:", ["%.4f" % v for v in hist[-6:]], "f=", f(hist[-1]))
print("first 8:", ["%.4f" % v for v in hist[:8]])
