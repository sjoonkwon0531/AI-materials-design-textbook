"""Lab 5.3 (53_ssa): Gillespie SSA vs deterministic ODE.
Part A: Ch.2 Lab 2.3 linear network X1 -> X2 <-> X3 (mean vs matrix exponential,
        fluctuations vs N, exact binomial stationary width).
Part B: stiffness reappears as event count (fast reversible X1 <-> X2).
Part C: Schloegl bistable model -- ODE picks one branch, SSA switches.
All seeds fixed; results vary slightly with seed.
"""
import time
import numpy as np
from numba import njit
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- SSA core
@njit(cache=True)
def propensities(x, k, a):
    # linear network: R0: X1->X2 (k0), R1: X2->X3 (k1), R2: X3->X2 (k2), R3: X2->X1 (k3)
    a[0] = k[0] * x[0]
    a[1] = k[1] * x[1]
    a[2] = k[2] * x[2]
    a[3] = k[3] * x[1]

@njit(cache=True)
def ssa_linear(x0, k, t_obs, seed):
    """Direct-method SSA (Gillespie 1976). Returns states at t_obs and #events."""
    np.random.seed(seed)
    nu = np.array([[-1, 1, 0], [0, -1, 1], [0, 1, -1], [1, -1, 0]])
    x = x0.copy()
    a = np.zeros(4)
    out = np.zeros((t_obs.size, 3))
    t = 0.0
    j = 0
    nev = 0
    while j < t_obs.size:
        propensities(x, k, a)
        R = a.sum()
        if R <= 0.0:
            tau = 1e300
        else:
            tau = -np.log(np.random.random()) / R       # Delta t = -ln u / R_tot
        # record observations that fall before the next event
        while j < t_obs.size and t + tau > t_obs[j]:
            out[j, :] = x
            j += 1
        if j >= t_obs.size:
            break
        # choose reaction mu with probability a_mu / R
        r = np.random.random() * R
        mu = 0
        c = a[0]
        while c < r:
            mu += 1
            c += a[mu]
        x += nu[mu]
        t += tau
        nev += 1
    return out, nev

@njit(cache=True)
def ssa_schlogl(x0, tmax, dt_rec, seed, s=4.0):
    """Schloegl model; the classic parameter set (c1=3e-7, c2=1e-4, c3*N2=200,
    c4=3.5, N1=1e5) with the system volume reduced by a factor s (fewer molecules,
    larger relative noise)."""
    np.random.seed(seed)
    c1, c2, c3, c4 = 3e-7 * s, 1e-4 * s * s, 1e-3 / s, 3.5
    N1, N2 = 1e5, 2e5
    x = x0
    t = 0.0
    nrec = int(tmax / dt_rec) + 1
    rec = np.zeros(nrec)
    j = 0
    nev = 0
    while j < nrec:
        a1 = c1 * N1 * x * (x - 1) / 2.0
        a2 = c2 * x * (x - 1) * (x - 2) / 6.0
        a3 = c3 * N2
        a4 = c4 * x
        R = a1 + a2 + a3 + a4
        tau = -np.log(np.random.random()) / R
        while j < nrec and t + tau > j * dt_rec:
            rec[j] = x
            j += 1
        r = np.random.random() * R
        if r < a1:
            x += 1
        elif r < a1 + a2:
            x -= 1
        elif r < a1 + a2 + a3:
            x += 1
        else:
            x -= 1
        t += tau
        nev += 1
    return rec, nev

def main():
    # ------------------------------------------------------------ Part A
    print("=== Part A: Ch.2 network X1 -(1001.5)-> X2 <-(0.5/1.0)-> X3 ===")
    k = np.array([1001.5, 0.5, 1.0, 0.0])
    A = np.array([[-k[0], 0, 0], [k[0], -k[1], k[2]], [0, k[1], -k[2]]])
    t_obs = np.array([0.01, 0.5, 1.0, 2.0, 4.0, 8.0])
    N0 = 1000
    R = 400
    X = np.zeros((R, t_obs.size, 3))
    nevs = []
    for r in range(R):
        X[r], ne = ssa_linear(np.array([N0, 0, 0]), k, t_obs, 1000 + r)
        nevs.append(ne)
    mean = X.mean(0) / N0
    sem = X.std(0, ddof=1) / N0 / np.sqrt(R)
    print(f"N0={N0}, trajectories={R}, mean #events per trajectory={np.mean(nevs):.0f}")
    print("   t    <x3>_SSA          exact(expm)   |diff|/sem")
    for i, t in enumerate(t_obs):
        ex = (expm(A * t) @ np.array([1.0, 0, 0]))[2]
        print(f"{t:5.2f}  {mean[i,2]:.4f} +/- {sem[i,2]:.4f}   {ex:.4f}       {abs(mean[i,2]-ex)/max(sem[i,2],1e-12):.2f}")
    # fluctuation scaling at t=8 (essentially stationary: binomial p=1/3)
    print("fluctuations of x3 = X3/N at t=8 (stationary binomial theory sqrt(p(1-p)/N), p=1/3):")
    stds = {}
    for N in [100, 1000, 10000]:
        vals = []
        for r in range(400):
            o, _ = ssa_linear(np.array([N, 0, 0]), k, np.array([8.0]), 5000 + r)
            vals.append(o[0, 2] / N)
        vals = np.array(vals)
        th = np.sqrt((1 / 3) * (2 / 3) / N)
        stds[N] = (vals.mean(), vals.std(ddof=1), th)
        print(f"  N={N:6d}: mean={vals.mean():.4f}  std={vals.std(ddof=1):.4f}  theory={th:.4f}")
    np.savez("../ch05/labs/53_ssa_partA.npz",
             t_obs=t_obs, X=X, N0=N0)
    # a few full trajectories for the figure
    tgrid = np.linspace(0.0, 8.0, 401)
    tgrid[0] = 1e-9
    trajs = {}
    for N in [100, 1000]:
        trajs[N] = np.array([ssa_linear(np.array([N, 0, 0]), k, tgrid, 9000 + r)[0][:, 2] / N
                             for r in range(5)])
    exact = np.array([(expm(A * t) @ np.array([1.0, 0, 0]))[2] for t in tgrid])

    # ------------------------------------------------------------ Part B
    print("=== Part B: stiffness as event count (X1 <-> X2 fast, X2 <-> X3 slow) ===")
    for kfast in [10.0, 100.0, 1000.0]:
        kB = np.array([kfast, 0.5, 1.0, kfast])
        AB = np.array([[-kB[0], kB[3], 0], [kB[0], -kB[1] - kB[3], kB[2]], [0, kB[1], -kB[2]]])
        ev = np.linalg.eigvals(AB)
        nz = np.sort(np.abs(ev[np.abs(ev) > 1e-9]))
        S = nz[-1] / nz[0]
        ne_list = []
        t0 = time.perf_counter()
        for r in range(20):
            _, ne = ssa_linear(np.array([1000, 0, 0]), kB, np.array([8.0]), 7000 + r)
            ne_list.append(ne)
        el = time.perf_counter() - t0
        f = lambda t, y: AB @ y
        sol = solve_ivp(f, (0, 8), [1.0, 0, 0], method="BDF", rtol=1e-6, atol=1e-9)
        print(f"  k_fast={kfast:7.1f}: stiffness S={S:8.1f}  SSA events/traj={np.mean(ne_list):9.0f}"
              f"  BDF f-evals={sol.nfev:4d}")

    # ------------------------------------------------------------ Part C
    print("=== Part C: Schloegl model (bistable), volume factor s=4 ===")
    sc = 4.0
    roots = np.sort(np.roots([-1e-4 * sc * sc / 6, 3e-7 * sc * 1e5 / 2, -3.5, 1e-3 * 2e5 / sc]).real)
    print("  deterministic fixed points:", np.round(roots, 1), "(low/high stable, middle unstable)")
    def rhs(t, y):
        x = y[0]
        return [3e-7 * sc * 1e5 / 2 * x * x - 1e-4 * sc * sc / 6 * x ** 3 + 2e2 / sc - 3.5 * x]
    for x0 in [roots[1] - 2, roots[1] + 2]:
        sol = solve_ivp(rhs, (0, 50), [x0], method="LSODA", rtol=1e-8)
        print(f"  ODE from x0={x0:.1f}: x(50)={sol.y[0,-1]:.1f}")
    tmax, dtr = 2000.0, 0.05
    ssa_schlogl(60.0, 1.0, dtr, 1, sc)               # compile
    t0 = time.perf_counter()
    rec, nev = ssa_schlogl(62.0, tmax, dtr, 42, sc)
    el = time.perf_counter() - t0
    # count transitions with hysteresis: low basin x<35, high basin x>100
    state, ntr = (0 if rec[0] < roots[1] else 1), 0
    for v in rec:
        if state == 0 and v > 100: state, ntr = 1, ntr + 1
        elif state == 1 and v < 35: state, ntr = 0, ntr + 1
    hi = rec > roots[1]
    print(f"  SSA from x0=62, t_max={tmax:.0f}, seed=42: events={nev}, wall={el:.2f} s")
    print(f"  fraction of time above unstable point={hi.mean():.3f}, basin-to-basin transitions={ntr}")
    print(f"  mean dwell time per visit={tmax/(ntr+1):.1f} (time units)")
    print(f"  mean on low branch={rec[~hi].mean():.1f}, mean on high branch={rec[hi].mean():.1f}")
    np.savez("../ch05/labs/53_ssa_fig.npz", tgrid=tgrid, exact=exact,
             tr100=trajs[100], tr1000=trajs[1000], rec=rec, dtr=dtr, roots=roots,
             stdN=np.array([[N, *stds[N]] for N in stds]))

if __name__ == "__main__":
    main()
