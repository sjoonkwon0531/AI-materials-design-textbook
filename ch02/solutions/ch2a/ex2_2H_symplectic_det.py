# Ex 2.2.H: one-step maps for harmonic oscillator, determinants, shadow invariant
import sympy as sp, numpy as np
h, w = sp.symbols('Delta_t omega', positive=True)
x, v = sp.symbols('x v')
x1 = x + h*v - h**2*w**2*x/2
v1 = v - h*w**2*(x + x1)/2
M = sp.Matrix([[sp.diff(x1,x), sp.diff(x1,v)],[sp.diff(v1,x), sp.diff(v1,v)]]).applyfunc(sp.expand)
print("M_VV =", M); print("det M_VV =", sp.simplify(M.det()), "; trace =", sp.expand(M.trace()))
ME = sp.Matrix([[1, h],[-h*w**2, 1]]); print("det M_EE =", sp.expand(ME.det()))
MI = sp.Matrix([[1, -h],[h*w**2, 1]]).inv(); print("det M_IE =", sp.simplify(MI.det()))
# shadow invariant: quadratic form Q = v^2 + w^2 (1 - h^2 w^2/4) x^2 is exactly conserved
Q = lambda X, V: V**2 + w**2*(1 - h**2*w**2/4)*X**2
print("Q(x1,v1) - Q(x,v) =", sp.simplify(sp.expand(Q(x1,v1) - Q(x,v))))
# consequence for Lab 2.2 (w=1,h=0.1,x0=1,v0=0): E = (v^2+x^2)/2 oscillates in band
hv = 0.1; xx, vv = 1.0, 0.0; Es = []
for n in range(int(2000/hv)):
    xn = xx + hv*vv - 0.5*hv*hv*xx; vv = vv - 0.5*hv*(xx+xn); xx = xn; Es.append(0.5*(vv*vv+xx*xx))
Es = np.array(Es)
print(f"Lab2.2 VV energy band over T=2000: [{Es.min():.6f}, {Es.max():.6f}], width {Es.max()-Es.min():.3e}; predicted width h^2/8 = {hv**2/8:.3e}")
