"""Ex 1.9.A: 1D Ising (h=0) transfer-matrix eigenvalues."""
import sympy as sp, numpy as np, itertools
b, J, h, lam = sp.symbols('beta J h lambda', positive=True)
T0 = sp.Matrix([[sp.exp(b*J), sp.exp(-b*J)], [sp.exp(-b*J), sp.exp(b*J)]])
ev = sp.solve(sp.det(T0 - lam*sp.eye(2)), lam)
print("1.9.A eigenvalues:", [sp.simplify(e.rewrite(sp.cosh)) for e in ev])
print("  check:", [sp.simplify((e - 2*sp.cosh(b*J)).rewrite(sp.exp)) for e in ev], [sp.simplify((e - 2*sp.sinh(b*J)).rewrite(sp.exp)) for e in ev])
