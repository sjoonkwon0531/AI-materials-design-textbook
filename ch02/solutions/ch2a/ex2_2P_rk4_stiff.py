# Ex 2.2.P(b): RK4 on the Lab 2.3 stiff network is bound to dt <= 2.785/|lam_max|
import numpy as np
from scipy.optimize import brentq
from stiffnet import A, X0, T
lam, V = np.linalg.eig(A); c = np.linalg.solve(V, X0); exact = lambda t: ((V*np.exp(lam*t))@c).real
g = lambda z: 1+z+z**2/2+z**3/6+z**4/24
xr = brentq(lambda x: abs(g(x))-1, -3.5, -2.0)
print(f"RK4 limit dt <= {abs(xr):.6f}/1001.5 = {abs(xr)/1001.5:.6f}  (explicit Euler: {2/1001.5:.6f})")
F = lambda X: A@X
for h in [0.0027, 0.00278, 0.00279, 0.0029]:
    n = int(round(T/h)); X = X0.copy()
    for _ in range(n):
        k1=F(X); k2=F(X+h/2*k1); k3=F(X+h/2*k2); k4=F(X+h*k3); X = X+h/6*(k1+2*k2+2*k3+k4)
    print(f"  dt={h}: |g(-1001.5 dt)|={abs(g(-1001.5*h)):.4f}, steps={n}, f-evals={4*n}, max|X(8)-exact|={np.abs(X-exact(n*h)).max():.2e}")
print(f"RK4 at the limit needs >= {4*int(np.ceil(T/(abs(xr)/1001.5)))} f-evals for t in [0,8]; BDF2 with dt=0.1 needs 80 linear solves")
