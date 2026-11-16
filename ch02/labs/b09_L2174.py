import numpy as np
from scipy.optimize import least_squares
H = 20                                          # hidden units (params=61)
def fwd(p, x):                                  # u_theta and u'' (analytic)
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
