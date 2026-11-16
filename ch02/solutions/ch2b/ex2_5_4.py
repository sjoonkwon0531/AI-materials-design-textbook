# Ex 2.5.4: explicit FTCS in d dimensions: g = 1 - 4 lam sum sin^2(theta_i/2), limit lam <= 1/(2d)
import numpy as np
rng = np.random.default_rng(0)
def ftcs(d, lam, n=32, steps=400):
    u = rng.standard_normal((n,)*d)*1e-3
    for _ in range(steps):
        lap = sum(np.roll(u, 1, a) + np.roll(u, -1, a) - 2*u for a in range(d))
        u = u + lam*lap
    return np.abs(u).max()
for d in [1, 2, 3]:
    lim = 1/(2*d); print(f"d={d}: theory lam_max = 1/(2d) = {lim:.4f};  g(theta=pi,...,pi) = 1-4 d lam")
    for lam in [0.95*lim, 1.05*lim]:
        print(f"    lam={lam:.4f}: g_min={1-4*d*lam:+.3f}  max|u| after 400 steps from 1e-3 noise = {ftcs(d, lam):.2e}")
# cost: for the same h, explicit dt_max = h^2/(2 d alpha); number of steps to reach t=1 with h=0.01
h = 0.01
for d in [1, 2, 3]:
    print(f"d={d}: dt_max = {h*h/(2*d):.2e}, steps to t=1: {int(np.ceil(1/(h*h/(2*d)))):,d}, unknowns: {int((1/h)**d):,d}, work ~ {int(np.ceil(1/(h*h/(2*d))))*int((1/h)**d):.1e}")
