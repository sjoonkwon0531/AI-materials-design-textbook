"""Ex 5.3.6 diagnostic: the Lab ZGB picks the reacting neighbour by a cyclic scan from a random start,
which is not uniform when several neighbours qualify (e.g. two adjacent O's: 3/4 vs 1/4).
Here: same ZGB with a uniform choice among qualifying neighbours, compared at y = 0.45, L = 64."""
import sys, importlib
import numpy as np
from numba import njit
sys.path.insert(0, "../ch05/labs")
zgb = importlib.import_module("53_lattice").zgb

@njit(cache=True)
def pick(s, i, j, L, target, di, dj):
    cand = np.empty(4, np.int64); nc = 0
    for d in range(4):
        if s[(i + di[d]) % L, (j + dj[d]) % L] == target:
            cand[nc] = d; nc += 1
    if nc == 0:
        return -1
    return cand[np.random.randint(nc)]

@njit(cache=True)
def zgb_uniform(L, y, n_warm, n_meas, seed):
    np.random.seed(seed)
    s = np.zeros((L, L), dtype=np.int8)
    di = np.array([1, -1, 0, 0]); dj = np.array([0, 0, 1, -1])
    nCO = 0; nO = 0; sumCO = 0.0; sumO = 0.0; nprod = 0
    for mcs in range(n_warm + n_meas):
        for trial in range(L * L):
            i = np.random.randint(L); j = np.random.randint(L)
            if np.random.random() < y:
                if s[i, j] != 0:
                    continue
                d = pick(s, i, j, L, 2, di, dj)
                if d >= 0:
                    s[(i + di[d]) % L, (j + dj[d]) % L] = 0; nO -= 1
                    if mcs >= n_warm: nprod += 1
                else:
                    s[i, j] = 1; nCO += 1
            else:
                d = np.random.randint(4)
                i2 = (i + di[d]) % L; j2 = (j + dj[d]) % L
                if s[i, j] != 0 or s[i2, j2] != 0:
                    continue
                s[i, j] = 2; s[i2, j2] = 2; nO += 2
                for (pi, pj) in ((i, j), (i2, j2)):
                    dd = pick(s, pi, pj, L, 1, di, dj)
                    if dd >= 0:
                        s[(pi + di[dd]) % L, (pj + dj[dd]) % L] = 0; s[pi, pj] = 0; nCO -= 1; nO -= 1
                        if mcs >= n_warm: nprod += 1
        if mcs >= n_warm:
            sumCO += nCO; sumO += nO
    N2 = L * L
    return sumCO / n_meas / N2, sumO / n_meas / N2, nprod / n_meas / N2

for y in (0.45, 0.50):
    a = np.array([zgb(64, y, 1500, 1500, sd)[:3] for sd in (1, 2, 3)])
    b = np.array([zgb_uniform(64, y, 1500, 1500, sd) for sd in (1, 2, 3)])
    for name, arr in (("ZGB cyclic-scan choice (Lab)", a), ("ZGB uniform choice          ", b)):
        m = arr.mean(0); e = arr.std(0, ddof=1) / np.sqrt(3)
        print(f"y={y:.2f} {name}: theta_O = {m[1]:.4f} +/- {e[1]:.4f}, theta_CO = {m[0]:.4f} +/- {e[0]:.4f}, R_CO2 = {m[2]:.4f} +/- {e[2]:.4f}")
