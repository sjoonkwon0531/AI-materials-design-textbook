"""Lab 2.15: minimal PINN for -u''=pi^2 sin(pi x), u(0)=u(1)=0.
tanh network with H=20 (61 params), 80 collocation points, BC weight sqrt(10),
seed=0, Levenberg-Marquardt (scipy least_squares method='lm')."""
import numpy as np
from scipy.optimize import least_squares
H = 20
def fwd(p, x):
    w,b,v,c = p[:H], p[H:2*H], p[2*H:3*H], p[3*H]
    a = np.tanh(np.outer(x,w)+b)
    return a@v + c,  (-2*a*(1-a**2)) @ (v*w**2)
xc = np.linspace(0,1,80); f = np.pi**2*np.sin(np.pi*xc)
def residual(p):
    u,uxx = fwd(p,xc)
    u0,_ = fwd(p,[0.]); u1,_ = fwd(p,[1.])
    return np.concatenate([-uxx-f, np.sqrt(10)*np.array([u0[0],u1[0]])])
p0 = 0.5*np.random.default_rng(0).standard_normal(3*H+1)   # initial params
sol = least_squares(residual, p0, method='lm')              # minimize physics residual
xf = np.linspace(0,1,1001); uf,_ = fwd(sol.x,xf)
err = np.abs(uf-np.sin(np.pi*xf))
print(f"hidden units H={H}, collocation pts={xc.size}, params={p0.size}")
print(f"status={sol.status} nfev={sol.nfev} ({sol.message})")
print(f"final residual cost (0.5 sum r^2) = {sol.cost:.3g}")
print(f"max |u_PINN - u_exact|            = {err.max():.3g}")
print(f"u_PINN(0)={fwd(sol.x,[0.])[0][0]:.2g}, u_PINN(1)={fwd(sol.x,[1.])[0][0]:.2g}")
# error at collocation pts only
uc,_=fwd(sol.x,xc); print("max err at collocation", np.abs(uc-np.sin(np.pi*xc)).max())
np.savez("lab2_15_data.npz", xf=xf, uf=uf, err=err, xc=xc, uc=uc)
