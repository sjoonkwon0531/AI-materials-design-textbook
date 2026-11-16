# Ex 2.6.3: explicit Euler (S=0, biharmonic explicit) blow-up limit for Lab 2.14 (N=256, L=4pi, eps=0.1)
import numpy as np
from ch_common import ch_run, grid
N, L, eps = 256, 4*np.pi, 0.10
dx = L/N; k1, KX, KY, K2 = grid(N, L)
K2max = K2.max(); kmax1 = np.abs(k1).max()
print(f"dx={dx:.5f}, k_max(1D)={kmax1:.1f}, |k|^2_max(2D corner)={K2max:.0f}")
# linear stability of forward Euler: g = 1 + dt*sigma(k), sigma = -k^2 f''(c) - eps^2 k^4 ; need dt <= 2/|sigma|max
for fpp, lab in [(-1.0, "c~0 (f''=-1)"), (2.0, "c~+-1 (f''=2)")]:
    sig = -K2*fpp - eps**2*K2**2
    print(f"theory {lab}: dt_max = 2/max|sigma| = {2/np.abs(sig).max():.3e}")
print(f"scaling forms: dx^4/eps^2 = {dx**4/eps**2:.3e};  dx^4/(pi^4 eps^2) = {dx**4/(np.pi**4*eps**2):.3e};  "
      f"2D corner 2/(eps^2 (2 pi^2/dx^2)^2) = dx^4/(2 pi^4 eps^2) = {dx**4/(2*np.pi**4*eps**2):.3e}")

def blows(dt, nsteps=4000, c0=None):
    state = {'bad': False}
    def cb(t, c, ch):
        if not np.isfinite(c).all() or np.abs(c).max() > 5: state['bad'] = True; return False
    with np.errstate(all='ignore'):
        ch_run(N=N, L=L, eps=eps, dt=dt, T=nsteps*dt, explicit=True, seed=1, c0=c0, callback=cb, every=50)
    return state['bad']

for start in ['noise (t=0)', 'phase-separated (t=40 state)']:
    c0 = None if start.startswith('noise') else ch_run(T=40.0, seed=1)
    lo, hi = 2.6e-6, 3.4e-6
    for _ in range(6):
        mid = 0.5*(lo+hi)
        if blows(mid, c0=c0): hi = mid
        else: lo = mid
    print(f"measured explicit limit from {start}: dt_crit in [{lo:.3e}, {hi:.3e}] (4000-step test)")
    dtc = lo
    print(f"   semi-implicit dt=0.05 / dt_crit = {0.05/dtc:,.0f}x ;  explicit steps to T=400: {400/dtc:.2e}")
