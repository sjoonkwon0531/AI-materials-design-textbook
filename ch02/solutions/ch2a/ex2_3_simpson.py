# Ex 2.3: RK4 with f=f(t) reduces to Simpson's rule
import numpy as np, sympy as sp
t, h, tn = sp.symbols('t h t_n')
f = sp.Function('f')
k1 = f(tn); k2 = f(tn+h/2); k3 = f(tn+h/2); k4 = f(tn+h)
inc = sp.simplify(h/6*(k1+2*k2+2*k3+k4))
print("RK4 increment:", inc)
# numeric: integrate f(t)=cos t on [0,1] with one RK4 step vs Simpson
fn = np.cos; a, H = 0.0, 1.0
def rk4_step(F, t0, y0, H):
    k1=F(t0,y0); k2=F(t0+H/2,y0+H/2*k1); k3=F(t0+H/2,y0+H/2*k2); k4=F(t0+H,y0+H*k3)
    return y0+H/6*(k1+2*k2+2*k3+k4)
rk = rk4_step(lambda tt,yy: fn(tt), a, 0.0, H)
simp = H/6*(fn(a)+4*fn(a+H/2)+fn(a+H))
print(f"RK4 one step = {rk:.15f}, Simpson = {simp:.15f}, diff = {abs(rk-simp):.1e}, exact sin(1) = {np.sin(1):.15f}")
# exact for cubics
p = lambda tt: 4*tt**3 - 3*tt**2 + 2*tt + 1
rkp = rk4_step(lambda tt,yy: p(tt), 0.0, 0.0, 2.0)
print(f"cubic on [0,2]: RK4 = {rkp:.12f}, exact = {2**4-2**3+2**2+2:.12f}")
