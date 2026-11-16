# Ex 2.2.J: nonlinear pendulum theta'' = -sin(theta), large amplitude theta0, VV vs RK4
import numpy as np
from scipy.special import ellipk
th0, h, T = 2.5, 0.1, 5000.0
Tp = 4*ellipk(np.sin(th0/2)**2)
print(f"theta0={th0} rad ({np.degrees(th0):.1f} deg), period={Tp:.4f} (small-angle 2pi=6.2832), T={T:g} -> {T/Tp:.1f} periods, dt={h}")
E = lambda th, w: 0.5*w*w - np.cos(th)
E0 = E(th0, 0.0); N = int(round(T/h))
def F(Y): return np.array([Y[1], -np.sin(Y[0])])
def rk4(Y):
    k1=F(Y); k2=F(Y+h/2*k1); k3=F(Y+h/2*k2); k4=F(Y+h*k3); return Y+h/6*(k1+2*k2+2*k3+k4)
def vv(Y):
    th, w = Y; a = -np.sin(th); thn = th + h*w + 0.5*h*h*a; wn = w + 0.5*h*(a - np.sin(thn)); return np.array([thn, wn])
res = {}
for name, st in [("RK4", rk4), ("velocity-Verlet", vv)]:
    Y = np.array([th0, 0.0]); Es = np.empty(N+1); Es[0]=E0; thmax = []
    for n in range(N):
        Y = st(Y); Es[n+1] = E(*Y)
    dE = (Es-E0)/abs(E0)
    # windowed band: first and last 10% of run
    m = N//10
    print(f"{name:16s} dE/|E0| first10%: [{dE[:m].min():+.3e},{dE[:m].max():+.3e}]  last10%: [{dE[-m:].min():+.3e},{dE[-m:].max():+.3e}]  final {dE[-1]:+.3e}")
    # turning-point amplitude: theta at w sign changes from + to -
    res[name] = Es
# amplitude decay for RK4: max |theta| in first vs last period (via energy: cos(theta_max) = -E)
for name, Es in res.items():
    amp_first = np.degrees(np.arccos(np.clip(-Es[:int(Tp/h)].mean(),-1,1)))
    amp_last  = np.degrees(np.arccos(np.clip(-Es[-int(Tp/h):].mean(),-1,1)))
    print(f"{name:16s} amplitude (from mean energy) first period {amp_first:.3f} deg, last period {amp_last:.3f} deg")
# smaller dt check for RK4 drift scaling
prevd = None
for hh in [0.2, 0.1, 0.05]:
    h = hh; N2 = int(round(1000/h)); Y = np.array([th0, 0.0])
    for n in range(N2): Y = rk4(Y)
    d = (E(*Y)-E0)/abs(E0)
    print(f"RK4 dt={hh}: rel. energy change at T=1000: {d:+.3e}" + ("" if prevd is None else f"  ratio={prevd/d:.1f} (p={np.log2(prevd/d):.2f})")); prevd = d
h = 0.1
