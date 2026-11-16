# Ex 2.6: y' = -2y + 3e^{-4t}, y(0)=1; RK4 vs explicit Euler convergence at T=5
import numpy as np
f = lambda t,y: -2*y + 3*np.exp(-4*t)
exact = lambda t: 2.5*np.exp(-2*t) - 1.5*np.exp(-4*t)
T = 5.0
def run(step, h):
    n = int(round(T/h)); y = 1.0; t = 0.0; emax = 0.0
    for i in range(n):
        y = step(t, y, h); t = (i+1)*h
        emax = max(emax, abs(y-exact(t)))
    return y, abs(y-exact(T)), emax
euler = lambda t,y,h: y + h*f(t,y)
def rk4(t,y,h):
    k1=f(t,y); k2=f(t+h/2,y+h/2*k1); k3=f(t+h/2,y+h/2*k2); k4=f(t+h,y+h*k3)
    return y + h/6*(k1+2*k2+2*k3+k4)
for name, st in [("Euler", euler), ("RK4", rk4)]:
    print(name)
    prev = None
    for h in [0.1, 0.05, 0.025, 0.0125]:
        y, eT, em = run(st, h)
        r = "" if prev is None else f"  ratio(T)={prev[0]/eT:5.2f} (p={np.log2(prev[0]/eT):.2f})  ratio(max)={prev[1]/em:5.2f} (p={np.log2(prev[1]/em):.2f})"
        print(f"  dt={h:<7} y(5)={y:.6e}  |err(5)|={eT:.2e}  max|err|={em:.2e}{r}")
        prev = (eT, em)
