# Ex 2.7.5: flux-form (conservative) neural update vs direct-output update, 1D periodic
#   flux form : c_i^{n+1} = c_i^n - dt/dx (J_{i+1/2} - J_{i-1/2}),  J_{i+1/2} = MLP_theta(c_{i-1}, c_i, c_{i+1}, c_{i+2})
#   direct    : c_i^{n+1} = c_i^n + dt * MLP_theta(c_{i-2..i+2})
import numpy as np
from scipy.optimize import least_squares
N, L, eps = 64, 4*np.pi, 0.2; dx = L/N          # coarse grid so that an explicit local update can be stable
k = np.fft.fftfreq(N, d=dx)*2*np.pi; K2 = k**2; K4 = K2**2
def ch1d(c, T, dt=1e-4, S=2.0, every=2e-4):
    den = 1 + dt*S*K2 + dt*eps**2*K4; gain = 1 + dt*S*K2; ch = np.fft.fft(c); out = [c.copy()]; ne = int(round(every/dt))
    for n in range(1, int(round(T/dt))+1):
        ch = (gain*ch - dt*K2*np.fft.fft(c**3 - c))/den; c = np.real(np.fft.ifft(ch))
        if n % ne == 0: out.append(c.copy())
    return np.array(out)

H = 8
def mlp(p, F, nin):                          # F: (..., nin) features -> scalar
    W1 = p[:nin*H].reshape(nin, H); b1 = p[nin*H:nin*H+H]; w2 = p[nin*H+H:nin*H+2*H]; b2 = p[-1]
    return np.tanh(F@W1 + b1)@w2 + b2
nparam = lambda nin: nin*H + 2*H + 1
def stencil(c, offs): return np.stack([np.roll(c, -o, axis=-1) for o in offs], axis=-1)
Dt = 2e-4                                       # surrogate step
print(f"grid dx={dx:.4f}; forward-Euler CH limit 2/(eps^2 kmax^4 + 2 kmax^2) = {2/(eps**2*np.abs(k).max()**4 + 2*np.abs(k).max()**2):.2e} > Dt={Dt}")
def step_flux(p, c):
    J = mlp(p, stencil(c, [-1, 0, 1, 2]), 4)          # J at face i+1/2
    return c - Dt/dx*(J - np.roll(J, 1, axis=-1))
def step_direct(p, c):
    return c + Dt*mlp(p, stencil(c, [-2, -1, 0, 1, 2]), 5)

# (1) untrained random weights: conservation is architectural, not learned
rng = np.random.default_rng(0); c = 0.1*(2*rng.random(N)-1) + 0.2
pf = 0.1*rng.standard_normal(nparam(4)); pd = 0.1*rng.standard_normal(nparam(5))
cf, cd = c.copy(), c.copy(); m0 = c.sum()*dx
for n in range(1000): cf = step_flux(pf, cf); cd = step_direct(pd, cd)
print(f"random weights, 1000 steps: |mass drift| flux-form = {abs(cf.sum()*dx-m0):.2e},  direct = {abs(cd.sum()*dx-m0):.2e}  (m0={m0:.4f})")

# (2) train both on 1D CH data (one-step map, Dt=2e-4, window t in [0.5,1.5])
X, Y = [], []
for s in range(6):
    tr = ch1d(0.1*(2*np.random.default_rng(10+s).random(N)-1), 1.5)[2500:]
    X.append(tr[:-1]); Y.append(tr[1:])
X = np.vstack(X); Y = np.vstack(Y)
idx = np.random.default_rng(1).choice(len(X), 120, replace=False); Xs, Ys = X[idx], Y[idx]
fits = {}
for name, stp, nin in [('flux', step_flux, 4), ('direct', step_direct, 5)]:
    sol = least_squares(lambda p: (stp(p, Xs) - Ys).ravel(), 0.1*np.random.default_rng(2).standard_normal(nparam(nin)),
                        method='trf', max_nfev=400)
    fits[name] = sol.x
    print(f"trained {name:6s}: params={nparam(nin)}, one-step train RMS = {np.sqrt(np.mean((stp(sol.x, Xs)-Ys)**2)):.3e}")
# rollout on an unseen trajectory
true = ch1d(0.1*(2*np.random.default_rng(99).random(N)-1), 2.0)[5000:]   # unseen quench, rollout t=1 -> 2 (5000 steps)
for name, stp in [('flux', step_flux), ('direct', step_direct)]:
    c = true[0].copy(); m0 = c.sum()*dx; drift = []; errs = []
    with np.errstate(all='ignore'):
        for n in range(1, len(true)):
            c = stp(fits[name], c); drift.append(abs(c.sum()*dx - m0)); errs.append(np.abs(c-true[n]).max())
    print(f"rollout {name:6s} ({len(true)-1} steps): mass drift at step 1000 = {drift[999]:.2e}, max over run = {np.nanmax(drift):.2e}; "
          f"final max|c| = {np.nanmax(np.abs(c)):.2e}, final max err vs CH = {np.nanmax(np.abs(c-true[-1])):.3e}")
    print(f"         error vs CH at steps 1/100/1000/5000: " + "/".join(f"{errs[i-1]:.2e}" for i in (1, 100, 1000, 5000)))
print(f"(true solver mass drift over the same window: {abs(true[-1].sum()-true[0].sum())*dx:.2e})")
