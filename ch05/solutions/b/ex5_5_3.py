"""Ex 5.5.3: Poiseuille slip of halfway bounce-back BGK; prediction u_s/u_max = (48 nu^2 - 1)/H^2."""
import sys, numpy as np
sys.path.insert(0, "../ch05/labs")
import importlib; lbm = importlib.import_module("55_lbm")
tau_star = 0.5 + np.sqrt(3/16)
print(f"tau* = 1/2 + sqrt(3/16) = {tau_star:.6f}")
for ny, tau in ((32, 0.6), (16, 0.6), (32, 0.8), (32, 1.2)):
    nu = (tau-0.5)/3; H = ny; umax = 0.01; g = 8*nu*umax/H**2
    y, u, ua, err, it, dr = lbm.poiseuille(ny, tau, g)
    us = np.mean(u-ua); spread = np.std(u-ua)/abs(us)
    pred = (48*nu**2-1)/H**2
    print(f"ny={ny} tau={tau}: predicted u_s/u_max={pred:+.6e}, measured={us/umax:+.6e}, "
          f"std/mean of (u-u_a)={spread:.1e}, L2 err={err:.3e}, iters={it}")
