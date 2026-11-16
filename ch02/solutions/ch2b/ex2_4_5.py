# Ex 2.4.5: P1 FEM on a non-uniform mesh (Lab 2.8 problem), vs uniform; FDM non-uniform stencil
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
exact = lambda x: np.sin(np.pi*x)
ffun = lambda x: np.pi**2*np.sin(np.pi*x)
gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])

def mesh(N, kind):
    s = np.linspace(0, 1, N+1)
    if kind == 'uniform': return s
    if kind == 'jitter':      # random +-30% node perturbation (non-smooth mesh)
        r = np.random.default_rng(0).uniform(-0.3, 0.3, N+1)/N; r[0] = r[-1] = 0; return s + r
    # clustered at the centre: x = s + a sin(2 pi s)/(2 pi), a=0.8 -> dx/ds = 1 + a cos(2 pi s)
    a = 0.8; return s + a*np.sin(2*np.pi*s)/(2*np.pi)

def fem(x):
    N = len(x)-1; he = np.diff(x)
    main = np.zeros(N+1); off = np.zeros(N); b = np.zeros(N+1)
    for e in range(N):                       # element stiffness (1/h_e)[[1,-1],[-1,1]]
        main[e] += 1/he[e]; main[e+1] += 1/he[e]; off[e] = -1/he[e]
        for xi in gp:
            xg = x[e] + 0.5*he[e]*(1+xi); jac = 0.5*he[e]
            b[e] += ffun(xg)*0.5*(1-xi)*jac; b[e+1] += ffun(xg)*0.5*(1+xi)*jac
    K = diags([off[1:N-1], main[1:N], off[1:N-1]], [-1, 0, 1], format='csr')
    u = np.zeros(N+1); u[1:N] = spsolve(K, b[1:N]); return u

def fdm(x):
    # non-uniform 3-point stencil: u'' ~ 2/(h_l+h_r) [ (u_{i+1}-u_i)/h_r - (u_i-u_{i-1})/h_l ]
    N = len(x)-1; h = np.diff(x); hl, hr = h[:-1], h[1:]
    lo = 2/(hl*(hl+hr)); up = 2/(hr*(hl+hr)); mid = -(lo+up)
    A = diags([lo[1:], mid, up[:-1]], [-1, 0, 1], format='csr')
    u = np.zeros(N+1); u[1:N] = spsolve(-A, ffun(x[1:N])); return u

def l2(x, u, nf=20):
    xf = np.concatenate([np.linspace(x[i], x[i+1], nf, endpoint=False) for i in range(len(x)-1)] + [[1.0]])
    return np.sqrt(np.trapezoid((np.interp(xf, x, u)-exact(xf))**2, xf))

print("   N | unif FEM L2  ord  | clust FEM L2 ord  | clust FEM nodal | clust FDM nodal ord | hmax/hmin | jitter FEM L2 | jitter FDM nodal")
prev = {}
for N in [8, 16, 32, 64, 128, 256]:
    xu, xn, xj = mesh(N, 'uniform'), mesh(N, 'cluster'), mesh(N, 'jitter')
    eu = l2(xu, fem(xu)); en = l2(xn, fem(xn))
    nn = np.abs(fem(xn)-exact(xn)).max(); fd = np.abs(fdm(xn)-exact(xn)).max()
    o = lambda k, v: f"{np.log2(prev[k]/v):5.2f}" if k in prev else "  -- "
    print(f"{N:4d} | {eu:.3e} {o('eu',eu)} | {en:.3e} {o('en',en)} | {nn:.2e}        | {fd:.3e} {o('fd',fd)} | {np.diff(xn).max()/np.diff(xn).min():6.2f}    | {l2(xj,fem(xj)):.3e}     | {np.abs(fdm(xj)-exact(xj)).max():.3e}")
    prev = dict(eu=eu, en=en, fd=fd)

# local truncation error of the non-uniform FDM stencil on the jittered mesh: (h_r-h_l)/3 u''' = O(h)
for N in [32, 128, 512]:
    x = mesh(N, 'jitter'); h = np.diff(x); hl, hr = h[:-1], h[1:]; xi = x[1:-1]
    u = exact(x); d2 = 2/(hl+hr)*((u[2:]-u[1:-1])/hr - (u[1:-1]-u[:-2])/hl)
    tau = np.abs(d2 - (-np.pi**2*np.sin(np.pi*xi))).max()
    print(f"jitter mesh N={N}: max local truncation error of FDM stencil = {tau:.3e}")
