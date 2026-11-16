# Ex 2.2.I: midpoint RK2 added to Lab 2.2 (m=k=1, x0=1, v0=0, dt=0.1), T=30 and T=2000 (318 periods)
import numpy as np
h = 0.1
F = lambda Y: np.array([Y[1], -Y[0]])
E = lambda Y: 0.5*(Y[0]**2 + Y[1]**2)
def euler(Y): return Y + h*F(Y)
def rk2(Y):  k1 = F(Y); k2 = F(Y + h/2*k1); return Y + h*k2
def rk4(Y):
    k1=F(Y); k2=F(Y+h/2*k1); k3=F(Y+h/2*k2); k4=F(Y+h*k3); return Y+h/6*(k1+2*k2+2*k3+k4)
def vv(Y):
    x, v = Y; xn = x + h*v - 0.5*h*h*x; vn = v - 0.5*h*(x + xn); return np.array([xn, vn])
for T in [30.0, 2000.0]:
    N = int(round(T/h)); print(f"T={T:g} ({T/(2*np.pi):.2f} periods), N={N}")
    for name, st in [("explicit Euler", euler), ("midpoint RK2", rk2), ("RK4", rk4), ("velocity-Verlet", vv)]:
        Y = np.array([1.0, 0.0]); Es = np.empty(N+1); Es[0] = E(Y)
        for n in range(N): Y = st(Y); Es[n+1] = E(Y)
        mono = np.all(np.diff(Es) >= 0) or np.all(np.diff(Es) <= 0)
        print(f"  {name:16s} E(T)={Es[-1]:.6g}  rel.drift={(Es[-1]-0.5)/0.5*100:+.4g}%  band=[{Es.min():.6f},{Es.max():.6f}]  monotone={mono}")
# per-step amplification |g(i h)| for each method
z = 1j*h
for name, g in [("Euler", 1+z), ("RK2", 1+z+z**2/2), ("RK4", 1+z+z**2/2+z**3/6+z**4/24)]:
    print(f"|g(i*0.1)|^2 {name}: {abs(g)**2:.12f}  -> predicted E factor over 20000 steps: {abs(g)**(2*20000):.6f}")
