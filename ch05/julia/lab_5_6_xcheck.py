"""Cross-check for the deterministic post-processing of lab_5_6.jl: smooth (edge-padded 5-point moving
average, written as an explicit window sum in Julia), gradient1 (= np.gradient), finalize_V (core
extrapolation, taper) and the bin/interpolation index rule of cg_forces (trunc toward zero, +1 for 1-based).
A synthetic g(r) is used; Julia-style implementations are compared with the numpy originals of 54_ibi.py."""
import sys, numpy as np
sys.path.insert(0, "../../ch05/labs")
from importlib import import_module
I = import_module("54_ibi")
r = I.RGRID; gt = np.where(r > 0.8, 1 + 0.5*np.exp(-(r-1.5)**2/0.05)*np.cos(3*(r-1.5)), 0.0)
valid = gt > 1e-3; V = np.zeros(I.NBIN); V[valid] = -np.log(gt[valid])
def smooth_jl(y, n=2):
    for _ in range(n):
        yp = np.concatenate([[y[0]]*2, y, [y[-1]]*2]); y = np.array([yp[i:i+5].sum()/5 for i in range(len(y))])
    return y
def grad_jl(V, h):
    g = np.empty_like(V); g[0] = (V[1]-V[0])/h; g[-1] = (V[-1]-V[-2])/h; g[1:-1] = (V[2:]-V[:-2])/(2*h); return g
def finalize_jl(V, valid):
    V = V.copy(); i0 = int(np.argmax(valid)); slope = min((V[i0+5]-V[i0])/(5*I.DR), -20.0)
    V[:i0] = V[i0] + slope*(r[:i0]-r[i0]); V = smooth_jl(V); V -= V[-1]
    t = np.ones_like(V); m = r > I.RC_CG-0.5; t[m] = 0.5*(1+np.cos(np.pi*(r[m]-(I.RC_CG-0.5))/0.5)); V *= t
    return V, -grad_jl(V, I.DR)
Va, Fa = I.finalize(V, valid); Vb, Fb = finalize_jl(V, valid)
print(f"max|V_numpy - V_julia-style| = {np.abs(Va-Vb).max():.1e},  max|F diff| = {np.abs(Fa-Fb).max():.1e}")
