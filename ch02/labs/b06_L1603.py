import numpy as np
def ftcs_step(u, lam):
    un = u.copy()
    un[1:-1] = u[1:-1] + lam*(u[2:] - 2*u[1:-1] + u[:-2])
    return un

h = 0.25; x = np.arange(0, 1+1e-9, h)
u0 = np.sin(np.pi*x); u0[0] = u0[-1] = 0          # IC = sin(pi x)

u_stable   = ftcs_step(u0, 0.4)    # k=0.025 -> lam=0.4
u_unstable = ftcs_step(u0, 4.0)    # k=0.25  -> lam=4
