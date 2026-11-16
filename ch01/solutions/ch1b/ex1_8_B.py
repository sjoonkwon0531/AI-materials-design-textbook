"""Ex 1.8.B: extended Dual class; Antoine dP*/dT by dual numbers vs PyTorch."""
import math, torch
class Dual:
    def __init__(self, real, dual=0.0): self.real, self.dual = real, dual
    @staticmethod
    def lift(o): return o if isinstance(o, Dual) else Dual(o)
    def __add__(self, o): o = Dual.lift(o); return Dual(self.real+o.real, self.dual+o.dual)
    __radd__ = __add__
    def __sub__(self, o): o = Dual.lift(o); return Dual(self.real-o.real, self.dual-o.dual)
    def __rsub__(self, o): return Dual.lift(o) - self
    def __neg__(self): return Dual(-self.real, -self.dual)
    def __mul__(self, o): o = Dual.lift(o); return Dual(self.real*o.real, self.real*o.dual + self.dual*o.real)
    __rmul__ = __mul__
    def __truediv__(self, o):
        o = Dual.lift(o); return Dual(self.real/o.real, (self.dual*o.real - self.real*o.dual)/o.real**2)
    def __rtruediv__(self, o): return Dual.lift(o)/self
    def __pow__(self, n):            # u^n, n real constant
        return Dual(self.real**n, n*self.real**(n-1)*self.dual)
    def __rpow__(self, c):           # c^u = exp(u ln c), c>0 constant
        return Dual(c**self.real, c**self.real*math.log(c)*self.dual)
    def __repr__(self): return f"{self.real} + {self.dual}ε"
def exp(u): return Dual(math.exp(u.real), math.exp(u.real)*u.dual) if isinstance(u, Dual) else math.exp(u)
def log(u): return Dual(math.log(u.real), u.dual/u.real) if isinstance(u, Dual) else math.log(u)
# unit tests against analytic derivatives at x=0.7
x = Dual(0.7, 1.0)
tests = {"x-3":   (x-3, 1.0), "1/x": (1/x, -1/0.7**2), "x^3.5": (x**3.5, 3.5*0.7**2.5),
         "exp(x)": (exp(x), math.exp(0.7)), "log(x)": (log(x), 1/0.7), "10^x": (10**x, 10**0.7*math.log(10)),
         "x/(1+x^2)": (x/(1+x**2), (1-0.7**2)/(1+0.7**2)**2)}
for k, (d, ref) in tests.items(): print(f"  d/dx {k:10s}: dual={d.dual:.15f}  exact={ref:.15f}  diff={abs(d.dual-ref):.1e}")
A, B, C = 8.07131, 1730.63, 233.426     # Ex 1.2.B constants (water; P* in mmHg, T in degC)
P_dual = lambda T: 10**(A - B/(C + T))
P_exact_dT = lambda T: 10**(A - B/(C+T))*math.log(10)*B/(C+T)**2
for T0 in [1.0, 25.0, 50.0, 100.0]:
    d = P_dual(Dual(T0, 1.0))
    Tt = torch.tensor(T0, dtype=torch.float64, requires_grad=True)
    Pt = 10**(A - B/(C + Tt)); Pt.backward()
    Tt32 = torch.tensor(T0, dtype=torch.float32, requires_grad=True)
    Pt32 = 10**(A - B/(C + Tt32)); Pt32.backward()
    ex = P_exact_dT(T0)
    print(f"T={T0:5.1f} C: P*={d.real:10.4f} mmHg, dP/dT dual={d.dual:.12f}, torch64={Tt.grad.item():.12f}, "
          f"exact={ex:.12f}, |dual-torch64|={abs(d.dual-Tt.grad.item()):.1e}, torch32 rel err={abs(Tt32.grad.item()/ex-1):.1e}")
