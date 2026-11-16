"""Cross-check for the deterministic part of lab_5_9.jl: tridiagonal backward-Euler FV exactly as in the
Julia file (I + dt L factored once; boundary faces doubled; steady state L u = bc), coarse() averaging of
consecutive 64-cell groups, trapezoid fluxes. Expected (chapter): homogenized 0.0488 0.0135 0.0056,
arithmetic 0.1603 0.1716 0.0056; fluxes 0.3981 0.3968 0.9102; macro 0.3967 (hom) 0.9100 (ari)."""
import numpy as np
from scipy.linalg import solve_banded
EPS = 1/64; AMP = 0.9; A = lambda x: 0.5 + x; a = lambda y: 1 + AMP*np.sin(2*np.pi*y); HARM = np.sqrt(1-AMP**2)
def fv(D, n, T=0.1, dt=1e-4, snaps=(0.01, 0.1)):
    h = 1/n; Dl = D[:-1].copy(); Dr = D[1:].copy(); Dl[0] *= 2; Dr[-1] *= 2
    dg = (Dl+Dr)/h**2; lo = -Dl[1:]/h**2; up = -Dr[:-1]/h**2
    ab = np.zeros((3, n)); ab[0, 1:] = dt*up; ab[1] = 1+dt*dg; ab[2, :-1] = dt*lo
    L = np.zeros((3, n)); L[0, 1:] = up; L[1] = dg; L[2, :-1] = lo
    bc = np.zeros(n); bc[0] = Dl[0]/h**2
    u = np.zeros(n); out = {}; ss = {int(round(s/dt)): s for s in snaps}
    for k in range(1, int(round(T/dt))+1):
        u = solve_banded((1, 1), ab, u+dt*bc)
        if k in ss: out[ss[k]] = u.copy()
    out['steady'] = solve_banded((1, 1), L, bc); return out
M = 32; xm = np.linspace(0, 1, M+1); nf = 2048; xf = np.linspace(0, 1, nf+1)
fine = fv(A(xf)*a(xf/EPS), nf)
coarse = lambda u: u.reshape(M, -1).mean(1)
for name, D in (("homogenized", A(xm)*HARM), ("arithmetic", A(xm))):
    r = fv(D, M); e = [np.abs(r[k]-coarse(fine[k])).max() for k in (0.01, 0.1, 'steady')]
    print(f"{name:12s} {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}   macro flux {D[0]*(1-r['steady'][0])/(0.5/M):.4f}")
xs = np.linspace(0, 1, 200001); tz = lambda y: np.sum((xs[1:]-xs[:-1])*(y[1:]+y[:-1]))/2
print(f"fluxes: fine {1/tz(1/(A(xs)*a(xs/EPS))):.4f} hom {1/tz(1/(A(xs)*HARM)):.4f} ari {1/tz(1/A(xs)):.4f}")
yb = (np.arange(16)+0.5)/16; print(f"lattice-exact harmonic (16 bonds) = {1/np.mean(1/a(yb)):.4f}")
