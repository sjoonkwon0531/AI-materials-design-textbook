"""Lab 56_hmm -- A minimal Heterogeneous Multiscale Method (HMM) for 1D diffusion.

Fine-scale truth:  du/dt = d/dx( D_eps(x) du/dx ),  D_eps(x) = A(x) * a(x/eps),
                   A(x) = 0.5 + x (slow),  a(y) = 1 + 0.9 sin(2 pi y) (fast), eps = 1/64.
Macro solver (HMM): finite volumes on M = 32 cells; the interface diffusivity is NOT given --
                   it is estimated on the fly by a micro solver: lattice random walkers (a KMC of
                   the random-barrier model, cf. Sec. 5.3) in one periodic micro cell with the slow
                   variable frozen at the interface position.
Compared with: fully resolved fine FV solution (2048 cells), exact homogenized D = A * <1/a>^-1,
               and the naive arithmetic average D = A * <a>.
Deterministic seeds. Runtime < 1 min.
"""
import numpy as np, time
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from numba import njit

EPS = 1/64; AMP = 0.9
A = lambda x: 0.5 + x
a = lambda y: 1 + AMP*np.sin(2*np.pi*y)
HARM = np.sqrt(1-AMP**2)            # <1/a>^{-1} for a = 1 + AMP sin(2 pi y)

# ------------------------------------------------------------------ FV diffusion solver
def fv_solve(D_face, n, T, dt, u_left=1.0, u_right=0.0, snaps=()):
    """Backward-Euler FV on [0,1], n cells, D given on the n+1 faces (Dirichlet at both ends)."""
    h = 1.0/n
    Dl = D_face[:-1].copy(); Dr = D_face[1:].copy()
    Dl[0] *= 2; Dr[-1] *= 2                    # boundary face: half-cell distance
    diag = (Dl+Dr)/h**2; lo = -Dl[1:]/h**2; up = -Dr[:-1]/h**2
    L = sp.diags([lo, diag, up], [-1, 0, 1], format='csc')
    M = sp.eye(n, format='csc') + dt*L
    lu = splu(M)
    bc = np.zeros(n); bc[0] = Dl[0]/h**2*u_left; bc[-1] = Dr[-1]/h**2*u_right
    u = np.zeros(n); out = {}; t = 0.0; nst = int(round(T/dt))
    snap_steps = {int(round(s/dt)): s for s in snaps}
    for k in range(1, nst+1):
        u = lu.solve(u + dt*bc); t += dt
        if k in snap_steps: out[snap_steps[k]] = u.copy()
    # steady state
    Lss = sp.diags([lo, diag, up], [-1, 0, 1], format='csc')
    out['steady'] = splu(Lss).solve(bc)
    return out

# ------------------------------------------------------------------ micro solver (KMC random walk)
@njit(cache=True)
def walk(w_bond, nwalk, nsteps, dtau, seed, nrec):
    """Random-barrier walk on a periodic ring of m sites; bond k joins site k and k+1.
    Returns mean-square displacement (in site units) recorded every nsteps//nrec steps."""
    np.random.seed(seed)
    m = w_bond.size
    pr = w_bond*dtau                            # prob. to hop right from site i
    pl = np.empty(m)                            # prob. to hop left  from site i
    for i in range(m): pl[i] = w_bond[(i-1) % m]*dtau
    every = nsteps//nrec; msd = np.zeros(nrec)
    for j in range(nwalk):
        i = np.random.randint(m)                # uniform start = stationary distribution
        pos = 0
        for s in range(1, nsteps+1):
            u = np.random.random()
            if u < pr[i]:
                pos += 1; i += 1
                if i == m: i = 0
            elif u < pr[i]+pl[i]:
                pos -= 1; i -= 1
                if i < 0: i = m-1
            if s % every == 0:
                msd[s//every-1] += pos*pos
    return msd/nwalk

def micro_D(x0, m=16, nwalk=4000, nsteps=8000, seed=0, nrec=80):
    """Estimate effective D at macro point x0: micro cell = one period eps, m lattice sites."""
    h = EPS/m
    y = (np.arange(m)+0.5)/m                   # bond midpoints in cell coordinates
    D_bond = A(x0)*a(y)                        # slow variable frozen at x0
    w = D_bond/h**2                            # hop rates
    dtau = 0.5/w.max()
    msd = walk(w, nwalk, nsteps, dtau, seed, nrec)*h**2
    t = dtau*(np.arange(1, nrec+1)*(nsteps//nrec))
    D_t = msd/(2*t)                            # running estimate
    half = nrec//2                             # fit slope on the second half (Fickian regime)
    slope = np.polyfit(t[half:], msd[half:], 1)[0]
    return slope/2, t, D_t, t[-1]*D_bond.mean()/EPS**2

if __name__ == "__main__":
    t0 = time.time()
    # ---------------- micro: short-time vs long-time diffusivity at one point
    x0 = 0.5
    D_hmm0, t_m, D_t, tau_cross = micro_D(x0, seed=1)
    print(f"[1] micro cell at x0={x0}: A={A(x0):.3f}, arithmetic A<a>={A(x0):.4f}, harmonic A<1/a>^-1={A(x0)*HARM:.4f}")
    print(f"    D(t)=MSD/2t at first record = {D_t[0]:.4f}, at last record = {D_t[-1]:.4f}")
    yb = (np.arange(16)+0.5)/16
    print(f"    lattice-exact harmonic mean (16 bonds) = {A(x0)/np.mean(1/a(yb)):.4f}")
    print(f"    slope estimate (second half) = {D_hmm0:.4f};  micro run length = {tau_cross:.1f} x eps^2/<D>")
    np.savez("../ch05/labs/56_micro.npz", t=t_m, D_t=D_t, arith=A(x0), harm=A(x0)*HARM)

    # seed-to-seed scatter of the micro estimator
    reps = np.array([micro_D(x0, seed=s)[0] for s in range(10, 20)])
    print(f"    10 independent micro runs: D = {reps.mean():.4f} +/- {reps.std(ddof=1):.4f} (std), exact {A(x0)*HARM:.4f}")

    # ---------------- HMM macro solve
    M = 32; xf_macro = np.linspace(0, 1, M+1)
    D_hmm = np.array([micro_D(x, seed=100+k)[0] for k, x in enumerate(xf_macro)])
    D_hom = A(xf_macro)*HARM
    D_ari = A(xf_macro)*1.0
    rel = (D_hmm-D_hom)/D_hom
    print(f"\n[2] HMM interface diffusivities (M={M}): mean |rel. error| vs homogenized = {np.abs(rel).mean():.3f}, max = {np.abs(rel).max():.3f}")

    T, dt = 0.1, 1e-4; snaps = (0.01, 0.1)
    # fine-scale truth
    n_f = 2048; xf_fine = np.linspace(0, 1, n_f+1)
    t1 = time.time()
    fine = fv_solve(A(xf_fine)*a(xf_fine/EPS), n_f, T, dt, snaps=snaps)
    t_fine = time.time()-t1
    xc_f = (np.arange(n_f)+0.5)/n_f; xc_m = (np.arange(M)+0.5)/M
    res = {}
    for name, D in (("HMM", D_hmm), ("homogenized", D_hom), ("arithmetic", D_ari)):
        res[name] = fv_solve(D, M, T, dt, snaps=snaps)
    # compare on macro cell centres (fine solution averaged over each macro cell)
    def coarse(u): return u.reshape(M, -1).mean(axis=1)
    print("\n[3] macro solution error vs fine-scale truth (cell averages, max norm)")
    print("   method         t=0.01     t=0.1      steady")
    for name in ("HMM", "homogenized", "arithmetic"):
        e = [np.abs(res[name][k]-coarse(fine[k])).max() for k in (0.01, 0.1, 'steady')]
        print(f"   {name:12s}  {e[0]:.4f}    {e[1]:.4f}    {e[2]:.4f}")
    # steady flux  J = -D du/dx  (exact for the fine problem: J = 1/int(1/D_eps))
    J_fine = 1/np.trapezoid(1/(A(np.linspace(0,1,200001))*a(np.linspace(0,1,200001)/EPS)), np.linspace(0,1,200001))
    J_hom = 1/np.trapezoid(1/(A(np.linspace(0,1,200001))*HARM), np.linspace(0,1,200001))
    J_ari = 1/np.trapezoid(1/A(np.linspace(0,1,200001)), np.linspace(0,1,200001))
    print(f"\n[4] steady flux: fine (exact quadrature) = {J_fine:.4f}, homogenized = {J_hom:.4f}, arithmetic = {J_ari:.4f}")
    h_m = 1.0/M
    for name, D in (("HMM", D_hmm), ("homogenized", D_hom), ("arithmetic", D_ari)):
        us = res[name]['steady']; J = D[0]*(1.0-us[0])/(h_m/2)
        print(f"    macro steady flux ({name}) = {J:.4f}")
    # cost accounting
    micro_ws = (M+1)*4000*8000
    print(f"\n[5] cost: fine FV {n_f} cells x {int(T/dt)} steps; HMM macro {M} cells x {int(T/dt)} steps "
          f"+ micro {micro_ws:.2e} walker-steps (once, coefficient is time-independent)")
    print(f"    fine-solver wall time {t_fine:.2f} s")
    np.savez("../ch05/labs/56_hmm.npz", xf_macro=xf_macro, D_hmm=D_hmm, D_hom=D_hom, D_ari=D_ari,
             xc_f=xc_f, xc_m=xc_m, fine01=fine[0.01], fine1=fine[0.1], fines=fine['steady'],
             hmm01=res["HMM"][0.01], hmm1=res["HMM"][0.1], hmms=res["HMM"]['steady'],
             ari01=res["arithmetic"][0.01], ari1=res["arithmetic"][0.1], aris=res["arithmetic"]['steady'])
    print(f"\nruntime {time.time()-t0:.1f} s")
