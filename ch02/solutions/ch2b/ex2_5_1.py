# Ex 2.5.1: hand check of FTCS step (h=0.25, IC sin(pi x)), lambda=0.4 and 0.6
import numpy as np
h = 0.25; x = np.arange(0, 1+1e-9, h); u0 = np.sin(np.pi*x); u0[0] = u0[-1] = 0
def step(u, lam):
    un = u.copy(); un[1:-1] = lam*u[:-2] + (1-2*lam)*u[1:-1] + lam*u[2:]; return un
print("IC:", np.round(u0, 4))
print("lam=0.4:", np.round(step(u0, 0.4), 4))
print("lam=0.6:", np.round(step(u0, 0.6), 4), " weights (lam,1-2lam,lam) =", (0.6, round(1-1.2, 2), 0.6))
th = np.array([1, 2, 3])*np.pi*h          # the 3 representable Dirichlet modes on this grid
for lam in [0.4, 0.6]:
    print(f"lam={lam}: g(theta=k pi h, k=1,2,3) =", np.round(1-4*lam*np.sin(th/2)**2, 4))
# pure sin(pi x) is an eigenvector -> decays; any other content (round-off) in mode k=3 grows by |g|=1.049
u = u0.copy()
for n in range(1, 1201):
    u = step(u, 0.6)
    if n in (10, 100, 400, 800, 1000, 1200): print(f"lam=0.6 step {n:4d}: max|u| = {np.abs(u).max():.3e}")
# discrete max principle violated: spike IC
spike = np.array([0, 0, 1.0, 0, 0])
print("spike IC, lam=0.6 one step:", step(spike, 0.6), " (centre < 0: max principle broken)")
print("spike IC, lam=0.4 one step:", step(spike, 0.4))
