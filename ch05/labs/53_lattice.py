"""Lab 5.3 (53_lattice): lattice kinetic Monte Carlo.
Part A: rejection-free (BKL/residence-time) KMC of adatom diffusion on a square
        lattice with bond-counting barriers E = Ea + n*Eb; Arrhenius check of D(T).
Part B: Ziff-Gulari-Barshad (1986) CO-oxidation model: kinetic phase diagram.
Seeds fixed; stochastic numbers vary slightly with seed.
"""
import time
import numpy as np
from numba import njit

KB = 8.617333262e-5  # eV/K

# ============================================================ Part A
@njit(cache=True)
def kmc_adatoms(L, N, Ea, Eb, nu, T, n_events, n_equil, seed, nrec=400):
    """Rejection-free KMC. Each adatom hops to an empty nearest-neighbour site with
    rate nu*exp(-(Ea + n_occ*Eb)/kT), n_occ = number of occupied neighbours of the
    *initial* site (bond counting). Unwrapped positions are recorded on a uniform
    time grid during the production run; D is fitted from MSD(lag) averaged over
    time origins and atoms. Returns D and the simulated production time."""
    np.random.seed(seed)
    beta = 1.0 / (KB * T)
    occ = -np.ones((L, L), dtype=np.int64)       # -1 empty, else atom index
    pos = np.zeros((N, 2), dtype=np.int64)
    unw = np.zeros((N, 2), dtype=np.int64)       # unwrapped displacement
    placed = 0
    while placed < N:
        i = np.random.randint(L); j = np.random.randint(L)
        if occ[i, j] < 0:
            occ[i, j] = placed; pos[placed, 0] = i; pos[placed, 1] = j; placed += 1
    # rate of a hop as a function of number of occupied neighbours (BKL classes)
    krate = np.zeros(5)
    for n in range(5):
        krate[n] = nu * np.exp(-beta * (Ea + n * Eb))
    di = np.array([1, -1, 0, 0]); dj = np.array([0, 0, 1, -1])
    rates = np.zeros(4 * N)
    t = 0.0
    t_meas0 = 0.0
    # expected production time from the initial total rate (grid spacing)
    rec = np.zeros((nrec, N, 2))
    trec = np.zeros(nrec)
    krec = 0
    dt_rec = 0.0
    for ev in range(n_equil + n_events):
        if ev == n_equil:
            t_meas0 = t
            for a in range(N):
                unw[a, 0] = 0; unw[a, 1] = 0
        # build event list (O(N) per step; production codes use class lists/trees)
        R = 0.0
        for a in range(N):
            i = pos[a, 0]; j = pos[a, 1]
            nocc = 0
            for d in range(4):
                if occ[(i + di[d]) % L, (j + dj[d]) % L] >= 0:
                    nocc += 1
            for d in range(4):
                if occ[(i + di[d]) % L, (j + dj[d]) % L] < 0:
                    rates[4 * a + d] = krate[nocc]
                else:
                    rates[4 * a + d] = 0.0
                R += rates[4 * a + d]
        # time increment and event selection
        if ev == n_equil:
            dt_rec = (n_events / R) / nrec        # grid from current mean residence time
        tau = -np.log(np.random.random()) / R
        if ev >= n_equil:
            while krec < nrec and t + tau > t_meas0 + krec * dt_rec:
                for a in range(N):
                    rec[krec, a, 0] = unw[a, 0]; rec[krec, a, 1] = unw[a, 1]
                trec[krec] = krec * dt_rec
                krec += 1
        t += tau
        r = np.random.random() * R
        c = 0.0
        e = 0
        while True:
            c += rates[e]
            if c >= r:
                break
            e += 1
        a = e // 4; d = e % 4
        i = pos[a, 0]; j = pos[a, 1]
        ni = (i + di[d]) % L; nj = (j + dj[d]) % L
        occ[i, j] = -1; occ[ni, nj] = a
        pos[a, 0] = ni; pos[a, 1] = nj
        unw[a, 0] += di[d]; unw[a, 1] += dj[d]
    # MSD(lag) averaged over time origins, lags up to krec/4
    nl = krec // 4
    lags = np.zeros(nl); msd = np.zeros(nl)
    for l in range(1, nl + 1):
        acc = 0.0; cnt = 0
        for o in range(krec - l):
            for a in range(N):
                dx = rec[o + l, a, 0] - rec[o, a, 0]; dy = rec[o + l, a, 1] - rec[o, a, 1]
                acc += dx * dx + dy * dy; cnt += 1
        lags[l - 1] = l * dt_rec; msd[l - 1] = acc / cnt
    # least-squares slope through the origin
    D = (lags * msd).sum() / (lags * lags).sum() / 4.0
    return D, t - t_meas0

# ============================================================ Part B
@njit(cache=True)
def zgb(L, y, n_warm, n_meas, seed):
    """ZGB model. 0 empty, 1 CO, 2 O. One MCS = L*L adsorption trials.
    Returns <theta_CO>, <theta_O>, CO2 production per site per MCS."""
    np.random.seed(seed)
    s = np.zeros((L, L), dtype=np.int8)
    di = np.array([1, -1, 0, 0]); dj = np.array([0, 0, 1, -1])
    cand = np.zeros(4, dtype=np.int64)
    nCO = 0; nO = 0
    sumCO = 0.0; sumO = 0.0; nprod = 0
    for mcs in range(n_warm + n_meas):
        for trial in range(L * L):
            i = np.random.randint(L); j = np.random.randint(L)
            if np.random.random() < y:                     # CO adsorption attempt
                if s[i, j] != 0:
                    continue
                # choose uniformly among the O neighbours (if any)
                nc = 0
                for d in range(4):
                    if s[(i + di[d]) % L, (j + dj[d]) % L] == 2:
                        cand[nc] = d; nc += 1
                reacted = False
                if nc > 0:
                    d = cand[np.random.randint(nc)]
                    ii = (i + di[d]) % L; jj = (j + dj[d]) % L
                    s[ii, jj] = 0; nO -= 1; reacted = True
                    if mcs >= n_warm:
                        nprod += 1
                if not reacted:
                    s[i, j] = 1; nCO += 1
            else:                                           # O2 dissociative attempt
                d = np.random.randint(4)
                i2 = (i + di[d]) % L; j2 = (j + dj[d]) % L
                if s[i, j] != 0 or s[i2, j2] != 0:
                    continue
                s[i, j] = 2; s[i2, j2] = 2; nO += 2
                for (pi, pj) in ((i, j), (i2, j2)):
                    nc = 0
                    for dd in range(4):
                        if s[(pi + di[dd]) % L, (pj + dj[dd]) % L] == 1:
                            cand[nc] = dd; nc += 1
                    if nc > 0:
                        dd = cand[np.random.randint(nc)]
                        ii = (pi + di[dd]) % L; jj = (pj + dj[dd]) % L
                        s[ii, jj] = 0; s[pi, pj] = 0; nCO -= 1; nO -= 1
                        if mcs >= n_warm:
                            nprod += 1
        if mcs >= n_warm:
            sumCO += nCO; sumO += nO
    N2 = L * L
    return sumCO / n_meas / N2, sumO / n_meas / N2, nprod / n_meas / N2, s

def main():
    # -------------------------------------------------------- Part A
    print("=== Part A: adatom diffusion, rejection-free KMC (BKL) ===")
    nu, Ea = 1.0e13, 0.50
    L = 40
    Ts = np.array([300.0, 350.0, 400.0, 500.0, 600.0, 800.0])
    # (i) single adatom: exact D = nu a^2 exp(-Ea/kT)
    print("(i) single adatom, L=40, 200 independent walkers x 20000 hops, a=1 (lattice units):")
    res1 = []
    for T in Ts:
        Dw = np.array([kmc_adatoms(L, 1, Ea, 0.0, nu, T, 20000, 0, 1000 + w, 200)[0] for w in range(200)])
        D = Dw.mean(); dD = Dw.std(ddof=1) / np.sqrt(Dw.size)
        Dex = nu * np.exp(-Ea / (KB * T))
        res1.append(D)
        print(f"  T={T:5.0f} K  D_KMC={D:.3e} +/- {dD:.1e}  D_exact={Dex:.3e}  ratio={D/Dex:.3f}")
    s1 = np.polyfit(1.0 / (KB * Ts), np.log(res1), 1)
    print(f"  Arrhenius fit: Ea_fit={-s1[0]:.3f} eV (input {Ea:.2f}), prefactor={np.exp(s1[1]):.2e} (input {nu:.1e})")
    # (ii) interacting lattice gas, coverage 0.10
    N = 160  # theta = 160/1600 = 0.10
    out = {}
    for Eb in [0.0, 0.10]:
        print(f"(ii) lattice gas L={L}, N={N} (theta=0.10), Eb={Eb:.2f} eV, 1e5 equil + 3e5 events, 3 seeds:")
        Ds = []
        for T in Ts:
            vals = []
            t0 = time.perf_counter()
            for sd in range(3):
                D, ttot = kmc_adatoms(L, N, Ea, Eb, nu, T, 300000, 100000, 100 + sd)
                vals.append(D)
            vals = np.array(vals)
            Ds.append(vals.mean())
            print(f"  T={T:5.0f} K  D={vals.mean():.3e} +/- {vals.std(ddof=1)/np.sqrt(3):.1e}"
                  f"  D/D_single={vals.mean()/(nu*np.exp(-Ea/(KB*T))):.3f}  t_sim={ttot:.2e} s"
                  f"  ({time.perf_counter()-t0:.1f} s wall)")
        Ds = np.array(Ds)
        sl = np.polyfit(1.0 / (KB * Ts), np.log(Ds), 1)
        hiT = Ts >= 500
        slh = np.polyfit(1.0 / (KB * Ts[hiT]), np.log(Ds[hiT]), 1)
        slL = np.polyfit(1.0 / (KB * Ts[~hiT]), np.log(Ds[~hiT]), 1)
        print(f"  apparent Ea (all T)={-sl[0]:.3f} eV; T>=500 K: {-slh[0]:.3f} eV; T<500 K: {-slL[0]:.3f} eV")
        out[Eb] = Ds
    # MD comparison: time reached at 300 K
    D, ttot = kmc_adatoms(L, N, Ea, 0.10, nu, 300.0, 300000, 100000, 100)
    print(f"  300 K, Eb=0.10: 3e5 production events cover t_sim={ttot:.2e} s = {ttot/1e-15:.1e} MD steps of 1 fs")

    # -------------------------------------------------------- Part B
    print("=== Part B: ZGB CO oxidation, L=128, 1500 warm-up + 1500 measured MCS, seed=1 ===")
    Lz = 128
    zgb(16, 0.5, 2, 2, 0)  # compile
    ys = np.unique(np.round(np.concatenate([[0.30, 0.34, 0.37], np.arange(0.38, 0.401, 0.005),
                                  [0.42, 0.44, 0.46, 0.48], np.arange(0.50, 0.541, 0.005),
                                  [0.56, 0.60]]), 3))
    t0 = time.perf_counter()
    rows = []
    for y in ys:
        tc, to, rp, _ = zgb(Lz, y, 1500, 1500, 1)
        rows.append((y, tc, to, rp))
        print(f"  y={y:.3f}  theta_CO={tc:.3f}  theta_O={to:.3f}  R_CO2={rp:.4f}")
    rows = np.array(rows)
    print(f"  wall time {time.perf_counter()-t0:.0f} s")
    act = rows[rows[:, 3] > 1e-3]
    print(f"  reactive window (R_CO2>1e-3): {act[0,0]:.3f} <= y <= {act[-1,0]:.3f}")
    imax = np.argmax(rows[:, 3])
    print(f"  max R_CO2={rows[imax,3]:.4f} at y={rows[imax,0]:.3f}")
    # snapshots for figure
    _, _, _, s1 = zgb(Lz, 0.45, 2000, 1, 1)
    _, _, _, s2 = zgb(Lz, 0.525, 300, 1, 3)
    np.savez("../ch05/labs/53_lattice_fig.npz", Ts=Ts, D0=out[0.0], D1=out[0.10],
             Dsingle=np.array(res1), nu=nu, Ea=Ea, zgb=rows, snap1=s1, snap2=s2)

if __name__ == "__main__":
    main()
