"""Lab 5.2 (key 52_ljmc): Lennard-Jones fluid Monte Carlo in NVT and NPT.

Reduced LJ units (epsilon = sigma = m = k_B = 1). N = 256 particles in a cubic box,
pair potential truncated at r_c = 2.5 with standard long-range (tail) corrections.

Part A  NVT isotherm T = 2.0: <U>/N and P versus the Thol et al. (2016) reference EOS
Part B  NPT at T = 2.0 with ln V moves and the (N+1) Jacobian factor: <rho> versus EOS,
        plus a deliberately wrong run that drops the Jacobian (N instead of N+1 -> 0 factor)
Part C  Same distribution, two roads: Metropolis MC vs Langevin (BAOAB) MD for the
        truncated-and-shifted potential at T = 2.0, rho = 0.6
Part D  Boundary conditions: periodic vs hard (reflecting) walls in z -> density profile

Runtime about 2-3 min on 2 cores. Deterministic seeds. Error bars by blocking (Flyvbjerg-Petersen).
"""
import time, json
import numpy as np
from numba import njit

RC = 2.5
RC2 = RC * RC

# --------------------------------------------------------------------------- utilities
def blocking_error(x):
    """Flyvbjerg-Petersen blocking: return (mean, plateau error, list of errors per level)."""
    x = np.asarray(x, float)
    errs = []
    while len(x) >= 32:
        n = len(x)
        errs.append(np.sqrt(np.var(x, ddof=1) / n))
        x = 0.5 * (x[: 2 * (n // 2) : 2] + x[1 : 2 * (n // 2) : 2])
    errs = np.array(errs)
    # plateau estimate: max over the levels that still have >= 128 blocks
    return errs, float(np.max(errs[: max(1, len(errs) - 2)]))


def u_tail(rho):
    return 8.0 / 3.0 * np.pi * rho * (1.0 / 3.0 * RC ** -9 - RC ** -3)


def p_tail(rho):
    return 16.0 / 3.0 * np.pi * rho ** 2 * (2.0 / 3.0 * RC ** -9 - RC ** -3)


def fcc_lattice(N, L):
    nc = int(round((N / 4) ** (1 / 3)))
    assert 4 * nc ** 3 == N
    a = L / nc
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = []
    for i in range(nc):
        for j in range(nc):
            for k in range(nc):
                for b in basis:
                    pts.append((np.array([i, j, k]) + b + 0.25) * a)
    return np.array(pts)


# --------------------------------------------------------------------------- energies
@njit(cache=True)
def pair_ev(r2, shift):
    """LJ pair energy and virial r.F for r2 < RC2. shift=1 -> truncated and shifted."""
    inv6 = 1.0 / (r2 * r2 * r2)
    e = 4.0 * inv6 * (inv6 - 1.0)
    if shift:
        ic6 = 1.0 / (RC2 * RC2 * RC2)
        e -= 4.0 * ic6 * (ic6 - 1.0)
    w = 24.0 * inv6 * (2.0 * inv6 - 1.0)
    return e, w


@njit(cache=True)
def particle_ev(pos, i, ri, L, Lz, walls, shift):
    N = pos.shape[0]
    e = 0.0
    w = 0.0
    for j in range(N):
        if j == i:
            continue
        dx = ri[0] - pos[j, 0]
        dy = ri[1] - pos[j, 1]
        dz = ri[2] - pos[j, 2]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        if not walls:
            dz -= Lz * np.round(dz / Lz)
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < RC2:
            a, b = pair_ev(r2, shift)
            e += a
            w += b
    return e, w


@njit(cache=True)
def total_ev(pos, L, Lz, walls, shift):
    N = pos.shape[0]
    e = 0.0
    w = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            if not walls:
                dz -= Lz * np.round(dz / Lz)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < RC2:
                a, b = pair_ev(r2, shift)
                e += a
                w += b
    return e, w


@njit(cache=True)
def nvt_sweep(pos, L, Lz, beta, delta, E, W, walls, shift):
    """N single-particle Metropolis trial displacements. Returns new E, W, number accepted."""
    N = pos.shape[0]
    nacc = 0
    ri = np.empty(3)
    for _ in range(N):
        i = np.random.randint(N)
        e_old, w_old = particle_ev(pos, i, pos[i], L, Lz, walls, shift)
        for d in range(3):
            ri[d] = pos[i, d] + delta * (2.0 * np.random.random() - 1.0)
        if walls:
            if ri[2] < 0.0 or ri[2] > Lz:     # hard wall: move rejected (reflecting boundary)
                continue
            ri[0] -= L * np.floor(ri[0] / L)
            ri[1] -= L * np.floor(ri[1] / L)
        else:
            for d in range(3):
                ri[d] -= L * np.floor(ri[d] / L)
        e_new, w_new = particle_ev(pos, i, ri, L, Lz, walls, shift)
        dE = e_new - e_old
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            for d in range(3):
                pos[i, d] = ri[d]
            E += dE
            W += w_new - w_old
            nacc += 1
    return E, W, nacc


@njit(cache=True)
def seed_numba(x):
    np.random.seed(x)


# --------------------------------------------------------------------------- Part A: NVT
def run_nvt(rho, T, N=256, n_eq=3000, n_prod=20000, seed=1, shift=0, return_series=False):
    np.random.seed(seed); seed_numba(seed)
    L = (N / rho) ** (1 / 3)
    pos = fcc_lattice(N, L)
    beta = 1.0 / T
    E, W = total_ev(pos, L, L, False, shift)
    delta = 0.1
    # equilibration with acceptance tuning toward 40 % (tuning is switched OFF in production)
    for s in range(n_eq):
        E, W, na = nvt_sweep(pos, L, L, beta, delta, E, W, False, shift)
        if (s + 1) % 50 == 0:
            acc = na / N
            delta *= 1.05 if acc > 0.40 else 0.95
            delta = min(delta, L / 2)
    E, W = total_ev(pos, L, L, False, shift)
    V = L ** 3
    us, ps = np.empty(n_prod), np.empty(n_prod)
    nacc_tot = 0
    for s in range(n_prod):
        E, W, na = nvt_sweep(pos, L, L, beta, delta, E, W, False, shift)
        nacc_tot += na
        us[s] = E / N
        ps[s] = rho * T + W / (3 * V)
    E2, _ = total_ev(pos, L, L, False, shift)
    drift = abs(E2 - E)
    out = dict(rho=rho, delta=delta, acc=nacc_tot / (N * n_prod), drift=drift)
    if not shift:
        us = us + u_tail(rho)
        ps = ps + p_tail(rho)
    out["u"], out["u_err"] = us.mean(), blocking_error(us)[1]
    out["p"], out["p_err"] = ps.mean(), blocking_error(ps)[1]
    if return_series:
        out["useries"] = us
    return out


# --------------------------------------------------------------------------- Part B: NPT
@njit(cache=True)
def scale_positions(pos, s):
    for i in range(pos.shape[0]):
        for d in range(3):
            pos[i, d] *= s


def run_npt(P, T, N=256, rho0=0.5, n_eq=4000, n_prod=20000, seed=2, jac=1):
    """NPT MC: one sweep = N displacement trials + 1 ln V trial.
    jac = 1 uses the correct (N+1) ln(V'/V) term; jac = 0 drops it entirely (wrong)."""
    np.random.seed(seed); seed_numba(seed)
    L = (N / rho0) ** (1 / 3)
    pos = fcc_lattice(N, L)
    beta = 1.0 / T
    E, W = total_ev(pos, L, L, False, 0)
    delta, dlnV = 0.1, 0.01
    nV_acc = 0
    rhos = np.empty(n_prod)
    for s in range(n_eq + n_prod):
        E, W, na = nvt_sweep(pos, L, L, beta, delta, E, W, False, 0)
        # ---- volume move in ln V
        V = L ** 3
        Etot_old = E + N * u_tail(N / V)
        lnVn = np.log(V) + dlnV * (2 * np.random.random() - 1)
        Vn = np.exp(lnVn)
        Ln = Vn ** (1 / 3)
        sfac = Ln / L
        scale_positions(pos, sfac)
        En, Wn = total_ev(pos, Ln, Ln, False, 0)
        Etot_new = En + N * u_tail(N / Vn)
        arg = -beta * (Etot_new - Etot_old + P * (Vn - V)) + jac * (N + 1) * np.log(Vn / V)
        if arg >= 0 or np.random.random() < np.exp(arg):
            L, E, W = Ln, En, Wn
            nV_acc += 1
            accV = True
        else:
            scale_positions(pos, 1.0 / sfac)
            accV = False
        if s < n_eq and (s + 1) % 50 == 0:
            delta *= 1.05 if na / N > 0.40 else 0.95
        if s < n_eq:
            dlnV *= 1.02 if accV else 0.99   # crude tuning during equilibration only
        if s == n_eq - 1:
            nV_acc = 0
        if s >= n_eq:
            rhos[s - n_eq] = N / L ** 3
    return dict(P=P, rho=rhos.mean(), rho_err=blocking_error(rhos)[1],
                accV=nV_acc / n_prod, dlnV=dlnV, series=rhos)


# --------------------------------------------------------------------------- Part C: Langevin MD
@njit(cache=True)
def forces_ts(pos, L):
    N = pos.shape[0]
    F = np.zeros_like(pos)
    E = 0.0
    ic6 = 1.0 / (RC2 * RC2 * RC2)
    eshift = 4.0 * ic6 * (ic6 - 1.0)
    for i in range(N - 1):
        for j in range(i + 1, N):
            d0 = pos[i, 0] - pos[j, 0]
            d1 = pos[i, 1] - pos[j, 1]
            d2 = pos[i, 2] - pos[j, 2]
            d0 -= L * np.round(d0 / L)
            d1 -= L * np.round(d1 / L)
            d2 -= L * np.round(d2 / L)
            r2 = d0 * d0 + d1 * d1 + d2 * d2
            if r2 < RC2:
                inv2 = 1.0 / r2
                inv6 = inv2 * inv2 * inv2
                E += 4.0 * inv6 * (inv6 - 1.0) - eshift
                fs = (48.0 * inv6 * inv6 - 24.0 * inv6) * inv2
                F[i, 0] += fs * d0; F[i, 1] += fs * d1; F[i, 2] += fs * d2
                F[j, 0] -= fs * d0; F[j, 1] -= fs * d1; F[j, 2] -= fs * d2
    return F, E


@njit(cache=True)
def baoab(pos, vel, L, T, gamma, dt, nsteps, every):
    """BAOAB Langevin integrator (Leimkuhler-Matthews). Returns potential energy per particle samples."""
    N = pos.shape[0]
    F, E = forces_ts(pos, L)
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1 * c1) * T)
    out = np.empty(nsteps // every)
    Ks = np.empty(nsteps // every)
    k = 0
    for s in range(nsteps):
        vel += 0.5 * dt * F                       # B
        pos += 0.5 * dt * vel                     # A
        for i in range(N):                        # O: exact Ornstein-Uhlenbeck step
            for d in range(3):
                vel[i, d] = c1 * vel[i, d] + c2 * np.random.standard_normal()
        pos += 0.5 * dt * vel                     # A
        for i in range(N):
            for d in range(3):
                pos[i, d] -= L * np.floor(pos[i, d] / L)
        F, E = forces_ts(pos, L)
        vel += 0.5 * dt * F                       # B
        if (s + 1) % every == 0:
            out[k] = E / N
            Ks[k] = 0.5 * np.sum(vel * vel)
            k += 1
    return out, Ks


def run_langevin(rho, T, N=256, dt=0.005, gamma=1.0, n_eq=4000, n_prod=40000, seed=3):
    np.random.seed(seed); seed_numba(seed)
    L = (N / rho) ** (1 / 3)
    pos = fcc_lattice(N, L)
    vel = np.random.normal(0, np.sqrt(T), (N, 3))
    baoab(pos, vel, L, T, gamma, dt, n_eq, 10)
    u, K = baoab(pos, vel, L, T, gamma, dt, n_prod, 10)
    Tkin = 2 * K / (3 * N)
    return dict(u=u.mean(), u_err=blocking_error(u)[1], Tkin=Tkin.mean(), Tkin_err=blocking_error(Tkin)[1])


# --------------------------------------------------------------------------- Part D: walls
def run_walls(rho, T, N=256, n_eq=3000, n_prod=10000, seed=4, nbins=120):
    """Periodic in x,y; hard walls at z=0 and z=Lz (MC moves crossing a wall are rejected).
    Box Lx = Ly = L, Lz = 2L so that the slab has a bulk-like middle."""
    np.random.seed(seed); seed_numba(seed)
    beta = 1.0 / T
    # choose L so that N/(L*L*Lz) = rho with Lz = 2L
    L = (N / (2 * rho)) ** (1 / 3)
    Lz = 2 * L
    nc = int(round((N / 4) ** (1 / 3)))
    pos = fcc_lattice(N, 1.0)              # unit cube fcc, rescale anisotropically
    pos[:, 0] *= L; pos[:, 1] *= L; pos[:, 2] *= Lz
    E, W = total_ev(pos, L, Lz, True, 0)
    delta = 0.1
    for s in range(n_eq):
        E, W, na = nvt_sweep(pos, L, Lz, beta, delta, E, W, True, 0)
        if (s + 1) % 50 == 0:
            delta *= 1.05 if na / N > 0.40 else 0.95
    hist = np.zeros(nbins)
    for s in range(n_prod):
        E, W, na = nvt_sweep(pos, L, Lz, beta, delta, E, W, True, 0)
        if s % 5 == 0:
            h, edges = np.histogram(pos[:, 2], bins=nbins, range=(0, Lz))
            hist += h
    nsamp = (n_prod + 4) // 5
    dz = Lz / nbins
    prof = hist / nsamp / (L * L * dz)
    zc = 0.5 * (edges[1:] + edges[:-1])
    return zc, prof, L, Lz


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    import teqp
    eos = teqp.make_model({"kind": "LJ126_TholJPCRD2016", "model": {}})
    z = np.array([1.0])

    def eos_up(T, rho):
        Ar01 = eos.get_Ar01(T, rho, z)
        Ar10 = eos.get_Ar10(T, rho, z)
        return T * Ar10, rho * T * (1 + Ar01)

    t0 = time.time()
    T = 2.0
    print("=== Part A: NVT MC, N=256, T*=2.0, r_c=2.5 + tail corrections, 3000 eq + 10000 prod sweeps ===")
    print("  rho    delta  acc   U/N (MC)          U/N (EOS)   P (MC)            P (EOS)")
    resA = []
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = run_nvt(rho, T, n_prod=10000)
        ue, pe = eos_up(T, rho)
        r["u_eos"], r["p_eos"] = ue, pe
        resA.append(r)
        print(f"  {rho:.1f}   {r['delta']:.3f}  {r['acc']:.2f}  {r['u']:+.4f} +/- {r['u_err']:.4f}  {ue:+.4f}     "
              f"{r['p']:.4f} +/- {r['p_err']:.4f}   {pe:.4f}")
    print(f"  (max |E_incremental - E_recomputed| at end = {max(r['drift'] for r in resA):.1e})")
    print("  finite-size check at rho=0.3 (2000 eq + 8000 prod sweeps, seed 11):")
    for Nfs in [108, 256, 500]:
        r = run_nvt(0.3, T, N=Nfs, n_eq=2000, n_prod=8000, seed=11)
        print(f"    N={Nfs:4d}: U/N = {r['u']:+.4f} +/- {r['u_err']:.4f}   P = {r['p']:.4f} +/- {r['p_err']:.4f}")
    print(f"  Part A time {time.time()-t0:.1f} s")

    # ---- Part B: NPT
    t1 = time.time()
    print("=== Part B: NPT MC, N=256, T*=2.0, one ln V trial per sweep, 3000 eq + 10000 prod sweeps ===")
    from scipy.optimize import brentq
    resB = []
    for P in [1.0, 3.0]:
        rho_eos = brentq(lambda r: eos_up(T, r)[1] - P, 0.05, 1.1)
        r = run_npt(P, T, n_eq=3000, n_prod=10000)
        r["rho_eos"] = rho_eos
        resB.append(r)
        print(f"  P={P:.1f}: <rho> = {r['rho']:.4f} +/- {r['rho_err']:.4f}   EOS rho = {rho_eos:.4f}   "
              f"acc(lnV) = {r['accV']:.2f}  dlnV = {r['dlnV']:.4f}")
    rw = run_npt(1.0, T, jac=0, seed=2, n_eq=3000, n_prod=10000)
    print(f"  P=1.0 WITHOUT the (N+1)ln(V'/V) term: <rho> = {rw['rho']:.4f} +/- {rw['rho_err']:.4f}  (wrong)")
    print(f"  Part B time {time.time()-t1:.1f} s")

    # ---- Part C: MC vs Langevin MD
    t2 = time.time()
    print("=== Part C: truncated-and-shifted LJ, T*=2.0, rho=0.6, N=256: Metropolis MC vs Langevin BAOAB MD ===")
    rmc = run_nvt(0.6, T, shift=1, seed=5, n_prod=10000)
    print(f"  MC  : <U_ts>/N = {rmc['u']:+.4f} +/- {rmc['u_err']:.4f}   (10000 sweeps)")
    md = {}
    for dt in [0.005]:
        rmd = run_langevin(0.6, T, dt=dt, n_prod=int(150 / dt), seed=6)
        md[dt] = rmd
        print(f"  MD  : <U_ts>/N = {rmd['u']:+.4f} +/- {rmd['u_err']:.4f}   kinetic T = {rmd['Tkin']:.4f} +/- {rmd['Tkin_err']:.4f}"
              f"   (BAOAB, gamma=1, dt={dt}, t_prod=150)")
    print(f"  Part C time {time.time()-t2:.1f} s")

    # ---- Part D: walls
    t3 = time.time()
    zc, prof, Lw, Lzw = run_walls(0.8, T, n_prod=6000)
    mid = (zc > 0.35 * Lzw) & (zc < 0.65 * Lzw)
    print("=== Part D: hard walls in z (reflecting: moves out of the box rejected), periodic in x,y; rho=0.8, T*=2.0 ===")
    print(f"  box {Lw:.3f} x {Lw:.3f} x {Lzw:.3f}; mid-slab density = {prof[mid].mean():.3f}; "
          f"first contact peak = {prof.max():.3f} at z = {zc[np.argmax(prof)]:.2f}")
    print(f"  Part D time {time.time()-t3:.1f} s;  total {time.time()-t0:.1f} s")

    # save for figures
    np.savez("../ch05/labs/52_ljmc_results.npz",
             rhoA=[r["rho"] for r in resA], uA=[r["u"] for r in resA], uA_err=[r["u_err"] for r in resA],
             pA=[r["p"] for r in resA], pA_err=[r["p_err"] for r in resA],
             rhoB_series1=resB[0]["series"], rhoB_series3=resB[1]["series"],
             rhoB_wrong=rw["series"], rhoB_eos=[r["rho_eos"] for r in resB],
             zc=zc, prof=prof, Lzw=Lzw)
    # dense EOS curve for plotting
    rr = np.linspace(0.02, 0.95, 120)
    up = np.array([eos_up(T, r) for r in rr])
    np.savez("../ch05/labs/52_ljmc_eos.npz", rho=rr, u=up[:, 0], p=up[:, 1])
