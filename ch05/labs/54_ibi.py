"""Lab 5.4 (54_ibi): iterative Boltzmann inversion (Reith, Puetz, Mueller-Plathe 2003).
Fine model : 2D fluid of rigid-ish LJ dimers (harmonic bond k=500, r0=0.8; inter-
             molecular LJ eps=sigma=1, rc=2.5), Langevin (BAOAB), T=1.0.
CG model   : one bead per dimer at the centre of mass, tabulated pair potential.
Steps      : (1) fine MD -> target g_CG(r), pressure; (2) PMF initial guess
             V0=-kT ln g; (3) IBI iterations; (4) pressure-corrected IBI;
             (5) representability/transferability: other density; (6) dynamics.
Reduced LJ units. Seeds fixed; numbers vary slightly with seed.
"""
import time
import numpy as np
from numba import njit

T = 1.0
NMOL = 200
RHO = 0.30          # molecules per unit area
KB_ = 500.0; R0 = 0.8
RC_LJ = 2.5
RC_CG = 3.5
DR = 0.01
NBIN = int(RC_CG / DR)
RGRID = (np.arange(NBIN) + 0.5) * DR

# ------------------------------------------------------------- fine model
@njit(cache=True)
def fine_forces(x, L, N):
    """x: (2N,2) atoms; atoms 2i,2i+1 form molecule i. Returns F, virial W=sum r.f."""
    n = 2 * N
    F = np.zeros_like(x)
    W = 0.0
    rc2 = RC_LJ * RC_LJ
    for i in range(n):
        for j in range(i + 1, n):
            if i // 2 == j // 2:
                continue
            dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]
            dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
            r2 = dx * dx + dy * dy
            if r2 < rc2:
                ir2 = 1.0 / r2; ir6 = ir2 * ir2 * ir2
                fr = (48.0 * ir6 * ir6 - 24.0 * ir6) * ir2
                F[i, 0] += fr * dx; F[i, 1] += fr * dy
                F[j, 0] -= fr * dx; F[j, 1] -= fr * dy
                W += fr * r2
    for m in range(N):
        i = 2 * m; j = i + 1
        dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        r = np.sqrt(dx * dx + dy * dy)
        fr = -KB_ * (r - R0) / r
        F[i, 0] += fr * dx; F[i, 1] += fr * dy
        F[j, 0] -= fr * dx; F[j, 1] -= fr * dy
        W += fr * r * r
    return F, W

@njit(cache=True)
def com(x, L, N):
    c = np.zeros((N, 2))
    for m in range(N):
        dx = x[2 * m + 1, 0] - x[2 * m, 0]; dy = x[2 * m + 1, 1] - x[2 * m, 1]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        c[m, 0] = x[2 * m, 0] + 0.5 * dx; c[m, 1] = x[2 * m, 1] + 0.5 * dy
    return c

@njit(cache=True)
def hist_add(c, L, N, h):
    for i in range(N):
        for j in range(i + 1, N):
            dx = c[i, 0] - c[j, 0]; dy = c[i, 1] - c[j, 1]
            dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
            r = np.sqrt(dx * dx + dy * dy)
            b = int(r / DR)
            if b < h.size:
                h[b] += 2.0

@njit(cache=True)
def run_fine(N, L, nequil, nprod, dt, gamma, seed, every):
    np.random.seed(seed)
    n = 2 * N
    # lattice initial positions of molecules, random orientation
    side = int(np.ceil(np.sqrt(N)))
    a = L / side
    x = np.zeros((n, 2))
    m = 0
    for ix in range(side):
        for iy in range(side):
            if m < N:
                th = 2 * np.pi * np.random.random()
                cx = (ix + 0.5) * a; cy = (iy + 0.5) * a
                x[2 * m, 0] = cx + 0.5 * R0 * np.cos(th); x[2 * m, 1] = cy + 0.5 * R0 * np.sin(th)
                x[2 * m + 1, 0] = cx - 0.5 * R0 * np.cos(th); x[2 * m + 1, 1] = cy - 0.5 * R0 * np.sin(th)
                m += 1
    v = np.random.normal(0.0, np.sqrt(T), (n, 2))
    F, W = fine_forces(x, L, N)
    c1 = np.exp(-gamma * dt); c2 = np.sqrt(1 - c1 * c1) * np.sqrt(T)
    h = np.zeros(NBIN)
    nfr = 0
    Wsum = 0.0; Ksum = 0.0
    unw = np.zeros((N, 2)); cprev = com(x, L, N)
    msd_t = np.zeros(nprod // every + 1)
    c0 = np.zeros((N, 2))
    for step in range(nequil + nprod):
        v += 0.5 * dt * F
        x += 0.5 * dt * v
        v = c1 * v + c2 * np.random.normal(0.0, 1.0, (n, 2))
        x += 0.5 * dt * v
        x %= L
        F, W = fine_forces(x, L, N)
        v += 0.5 * dt * F
        if step >= nequil:
            c = com(x, L, N)
            d = c - cprev
            d -= L * np.round(d / L)
            unw += d
            cprev = c
            k = step - nequil
            if k % every == 0:
                hist_add(c, L, N, h)
                nfr += 1
                Wsum += W
                Ksum += 0.5 * (v * v).sum()
                if k == 0:
                    c0[:, :] = unw
                msd_t[k // every] = ((unw - c0) ** 2).sum() / N
        else:
            cprev = com(x, L, N)
    A = L * L
    Kmean = Ksum / nfr
    # 2D virial pressure with atomic virial (kinetic part from measured K)
    P = (Kmean + 0.5 * Wsum / nfr) / A
    return h, nfr, P, msd_t

# ------------------------------------------------------------- CG model
@njit(cache=True)
def cg_forces(c, L, N, Ftab):
    F = np.zeros_like(c)
    W = 0.0
    nb = Ftab.size
    for i in range(N):
        for j in range(i + 1, N):
            dx = c[i, 0] - c[j, 0]; dy = c[i, 1] - c[j, 1]
            dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
            r = np.sqrt(dx * dx + dy * dy)
            s = r / DR - 0.5
            b = int(s)
            if b < 0:
                f = Ftab[0]
            elif b >= nb - 1:
                continue
            else:
                w = s - b
                f = (1 - w) * Ftab[b] + w * Ftab[b + 1]
            fr = f / r
            F[i, 0] += fr * dx; F[i, 1] += fr * dy
            F[j, 0] -= fr * dx; F[j, 1] -= fr * dy
            W += f * r
    return F, W

@njit(cache=True)
def run_cg(N, L, Ftab, nequil, nprod, dt, gamma, seed, every, mass):
    np.random.seed(seed)
    side = int(np.ceil(np.sqrt(N)))
    a = L / side
    c = np.zeros((N, 2))
    m = 0
    for ix in range(side):
        for iy in range(side):
            if m < N:
                c[m, 0] = (ix + 0.5) * a; c[m, 1] = (iy + 0.5) * a
                m += 1
    v = np.random.normal(0.0, np.sqrt(T / mass), (N, 2))
    F, W = cg_forces(c, L, N, Ftab)
    c1 = np.exp(-gamma * dt); c2 = np.sqrt(1 - c1 * c1) * np.sqrt(T / mass)
    h = np.zeros(NBIN)
    nfr = 0
    Wsum = 0.0
    unw = np.zeros((N, 2))
    msd_t = np.zeros(nprod // every + 1)
    for step in range(nequil + nprod):
        v += 0.5 * dt * F / mass
        dxh = 0.5 * dt * v
        c += dxh
        v = c1 * v + c2 * np.random.normal(0.0, 1.0, (N, 2))
        dx2 = 0.5 * dt * v
        c += dx2
        if step >= nequil:
            unw += dxh + dx2
        c %= L
        F, W = cg_forces(c, L, N, Ftab)
        v += 0.5 * dt * F / mass
        if step >= nequil:
            k = step - nequil
            if k % every == 0:
                hist_add(c, L, N, h)
                nfr += 1
                Wsum += W
                msd_t[k // every] = (unw ** 2).sum() / N
    A = L * L
    P = (N * T + 0.5 * Wsum / nfr) / A   # equipartition kinetic term
    return h, nfr, P, msd_t

# ------------------------------------------------------------- helpers
def gofr(h, nfr, N, L):
    rho = N / (L * L)
    shell = np.pi * ((RGRID + 0.5 * DR) ** 2 - (RGRID - 0.5 * DR) ** 2)
    return h / (nfr * N * rho * shell)

def smooth(y, n=2):
    for _ in range(n):
        y = np.convolve(np.pad(y, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return y

def finalize(V, valid):
    """Extrapolate core, smooth, shift and taper to zero at RC_CG; return V, Ftab."""
    V = V.copy()
    i0 = np.argmax(valid)
    slope = (V[i0 + 5] - V[i0]) / (5 * DR)
    slope = min(slope, -20.0)
    V[:i0] = V[i0] + slope * (RGRID[:i0] - RGRID[i0])
    V = smooth(V, 2)
    V -= V[-1]
    taper = np.ones_like(V)
    m = RGRID > RC_CG - 0.5
    taper[m] = 0.5 * (1 + np.cos(np.pi * (RGRID[m] - (RC_CG - 0.5)) / 0.5))
    V *= taper
    Ftab = -np.gradient(V, DR)
    return V, Ftab

def gerr(g, gt):
    m = RGRID > 0.5
    return np.sqrt(np.mean((g[m] - gt[m]) ** 2))

def main():
    L = np.sqrt(NMOL / RHO)
    dt = 0.005
    print(f"=== Step 1: fine MD, {NMOL} LJ dimers (bond k={KB_:.0f}, r0={R0}), rho_mol={RHO}, "
          f"L={L:.2f}, T={T}, dt={dt}, gamma=1.0 ===")
    t0 = time.perf_counter()
    h, nfr, P_fine, _ = run_fine(NMOL, L, 15000, 80000, dt, 1.0, 1, 40)
    g_t = gofr(h, nfr, NMOL, L)
    print(f"  frames={nfr}, P_fine={P_fine:.4f}, wall={time.perf_counter()-t0:.0f} s")
    ipk = np.argmax(g_t)
    print(f"  target g_CG(r): first peak r={RGRID[ipk]:.3f}, g={g_t[ipk]:.3f}")

    valid = g_t > 1e-3
    V = np.zeros(NBIN)
    V[valid] = -T * np.log(g_t[valid])
    V, Ftab = finalize(V, valid)
    hist = []
    Vs = [V.copy()]
    gs = []
    print("=== Step 2-3: IBI, V_{n+1} = V_n + alpha kT ln(g_n/g_target), alpha=1, CG MD 5k+30k steps ===")
    print("  iter   g RMSE    P_CG      (P_fine={:.4f})".format(P_fine))
    for it in range(9):
        hc, nf, Pcg, _ = run_cg(NMOL, L, Ftab, 5000, 30000, dt, 1.0, 100 + it, 15, 2.0)
        g = gofr(hc, nf, NMOL, L)
        e = gerr(g, g_t)
        hist.append((it, e, Pcg)); gs.append(g.copy())
        print(f"  {it:3d}   {e:.4f}   {Pcg:.4f}")
        if it == 8:
            break
        m = valid & (g > 1e-3)
        V[m] += 1.0 * T * np.log(g[m] / g_t[m])
        V, Ftab = finalize(V, valid)
        Vs.append(V.copy())
    # statistical noise floor: same final potential, independent seed
    hc, nf, _, _ = run_cg(NMOL, L, Ftab, 5000, 30000, dt, 1.0, 999, 15, 2.0)
    print(f"  noise floor (two independent CG runs, same V): g RMSE={gerr(gofr(hc, nf, NMOL, L), gs[-1]):.4f}")
    V_ibi = V.copy(); F_ibi = Ftab.copy(); P_ibi = hist[-1][2]; g_ibi = gs[-1]

    print("=== Step 4: pressure-corrected IBI, dV = A (1 - r/rc), A = -0.1 kT sign(dP) min(1, 3|dP|) ===")
    Vp = V.copy(); Fp = Ftab.copy()
    hp = []
    for it in range(7):
        hc, nf, Pcg, _ = run_cg(NMOL, L, Fp, 5000, 30000, dt, 1.0, 300 + it, 15, 2.0)
        g = gofr(hc, nf, NMOL, L)
        e = gerr(g, g_t)
        hp.append((it, e, Pcg))
        print(f"  {it:3d}   g RMSE={e:.4f}   P_CG={Pcg:.4f}")
        if it == 6:
            break
        dP = Pcg - P_fine
        Aamp = -0.1 * T * np.sign(dP) * min(1.0, 3.0 * abs(dP))
        m = valid & (g > 1e-3)
        Vp[m] += 1.0 * T * np.log(g[m] / g_t[m])
        Vp += Aamp * (1 - RGRID / RC_CG)
        Vp, Fp = finalize(Vp, valid)
    g_pc = g

    print("=== Step 5: transferability, same potentials at rho_mol=0.20 ===")
    L2 = np.sqrt(NMOL / 0.20)
    h2, n2, P2f, _ = run_fine(NMOL, L2, 10000, 50000, dt, 1.0, 2, 40)
    g2f = gofr(h2, n2, NMOL, L2)
    out = {}
    for name, Ft in [("IBI", F_ibi), ("IBI+P", Fp)]:
        hc, nf, Pc, _ = run_cg(NMOL, L2, Ft, 5000, 30000, dt, 1.0, 500, 15, 2.0)
        g2 = gofr(hc, nf, NMOL, L2)
        out[name] = g2
        print(f"  {name:6s}: g RMSE vs fine={gerr(g2, g2f):.4f}, P_CG={Pc:.4f}, P_fine={P2f:.4f}")

    print("=== Step 6: dynamics, weak Langevin gamma=0.05, rho_mol=0.30, 3 seeds x 30k steps ===")
    every = 50
    Dfs, Dcs = [], []
    for sd in range(3):
        _, _, _, msd_f = run_fine(NMOL, L, 10000, 30000, dt, 0.05, 3 + sd, every)
        _, _, _, msd_c = run_cg(NMOL, L, F_ibi, 10000, 30000, dt, 0.05, 3 + sd, every, 2.0)
        tt = np.arange(msd_f.size) * every * dt
        sel = tt > 0.25 * tt[-1]
        Dfs.append(np.polyfit(tt[sel], msd_f[sel], 1)[0] / 4)
        Dcs.append(np.polyfit(tt[sel], msd_c[sel], 1)[0] / 4)
    Dfs, Dcs = np.array(Dfs), np.array(Dcs)
    Df, Dc = Dfs.mean(), Dcs.mean()
    print("  per-seed D_fine:", np.round(Dfs, 3), " D_CG:", np.round(Dcs, 3))
    t0 = time.perf_counter(); run_fine(NMOL, L, 0, 2000, dt, 1.0, 4, 40); tf = time.perf_counter() - t0
    t0 = time.perf_counter(); run_cg(NMOL, L, F_ibi, 0, 2000, dt, 1.0, 4, 15, 2.0); tc = time.perf_counter() - t0
    print(f"  cost per step: fine={tf/2000*1e3:.3f} ms, CG={tc/2000*1e3:.3f} ms, ratio={tf/tc:.1f}")
    print(f"  D_COM(fine)={Df:.3f} +/- {Dfs.std(ddof=1)/np.sqrt(3):.3f}, D(CG, IBI)={Dc:.3f} +/- {Dcs.std(ddof=1)/np.sqrt(3):.3f},"
          f" speed-up D_CG/D_fine={Dc/Df:.2f}")
    np.savez("../ch05/labs/54_ibi_fig.npz", r=RGRID, g_t=g_t, gs=np.array(gs),
             Vs=np.array(Vs), hist=np.array(hist), hp=np.array(hp), P_fine=P_fine,
             g_pc=g_pc, Vp=Vp, g2f=g2f, g2_ibi=out["IBI"], g2_p=out["IBI+P"])

if __name__ == "__main__":
    main()
