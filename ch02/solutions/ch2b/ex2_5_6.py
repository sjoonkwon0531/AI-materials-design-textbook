# Ex 2.5.6: FDTD (Yee) CFL: S = c dt/dx <= 1/sqrt(d); high-index media
import numpy as np

def yee1d(S, nprof, steps=4000, N=400):
    # normalized units c=1, dx=1, dt=S. E at integer, H at half-integer nodes; eps_r = n^2
    E = np.zeros(N); H = np.zeros(N-1); eps = nprof**2
    x = np.arange(N); E += np.exp(-((x-100)/8.0)**2)             # smooth pulse in vacuum
    for _ in range(steps):
        H += S*(E[1:] - E[:-1])
        E[1:-1] += S/eps[1:-1]*(H[1:] - H[:-1])                  # PEC walls E=0 at ends
        if not np.isfinite(E).all() or np.abs(E).max() > 1e6: return np.inf
    return np.abs(E).max()

N = 400
vac = np.ones(N); slab = np.ones(N); slab[200:300] = 3.5          # Si-like slab, n = 3.5
with np.errstate(all='ignore'):
    for S in [0.5, 1.0, 1.01, 1.05]:
        print(f"1D S={S:4.2f}: vacuum max|E|={yee1d(S, vac):.3e}   with n=3.5 slab max|E|={yee1d(S, slab):.3e}")
    # the 'local' CFL in the slab would allow S = 3.5 there, but the vacuum cells dictate the global dt:
    print(f"1D S=3.50 (slab-local CFL) with slab: max|E|={yee1d(3.5, slab):.3e}")
    print(f"1D S=3.50 in uniform n=3.5 medium   : max|E|={yee1d(3.5, 3.5*np.ones(N)):.3e}  (local CFL c/n dt/dx<=1 holds)")
    print(f"1D S=3.60 in uniform n=3.5 medium   : max|E|={yee1d(3.6, 3.5*np.ones(N)):.3e}")

def yee2d(S, N=128, steps=1500):
    Ez = np.zeros((N, N)); Hx = np.zeros((N, N-1)); Hy = np.zeros((N-1, N))
    X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij'); Ez += np.exp(-((X-64)**2+(Y-64)**2)/30.)
    for _ in range(steps):
        Hx -= S*(Ez[:, 1:] - Ez[:, :-1]); Hy += S*(Ez[1:, :] - Ez[:-1, :])
        Ez[1:-1, 1:-1] += S*((Hy[1:, 1:-1] - Hy[:-1, 1:-1]) - (Hx[1:-1, 1:] - Hx[1:-1, :-1]))
        if not np.isfinite(Ez).all() or np.abs(Ez).max() > 1e6: return np.inf
    return np.abs(Ez).max()
with np.errstate(all='ignore'):
    for S in [0.70, 0.7071, 0.72]:
        print(f"2D S={S:6.4f} (limit 1/sqrt2={1/np.sqrt(2):.4f}): max|Ez|={yee2d(S):.3e}")

# cost model: resolve lambda0/n with P cells, global dt fixed by vacuum CFL dt = dx/(c sqrt d)
print("\ncost for fixed physical size & time, P = 20 cells per wavelength in the densest medium:")
for d in [1, 2, 3]:
    for n in [1.0, 2.0, 3.5]:
        rel = n**(d+1)            # cells ~ n^d, steps ~ n (dt ~ dx ~ 1/n)
        print(f"  d={d} n_max={n:3.1f}: work relative to n=1 -> {rel:8.1f}x")
