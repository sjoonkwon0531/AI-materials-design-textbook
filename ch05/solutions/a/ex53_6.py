"""Ex 5.3.6: ZGB as rejection-free KMC with physical time.
Rates: CO adsorption y per empty site; O2 adsorption (1-y)/2 per empty nearest-neighbour pair;
reaction k_r per adjacent CO-O pair. Three event classes kept in O(1) index lists (BKL classes).
Compared with the original ZGB (rejection, instantaneous reaction) at the same L = 64."""
import sys, importlib, time
import numpy as np
from numba import njit
sys.path.insert(0, "../ch05/labs")
zgb = importlib.import_module("53_lattice").zgb

@njit(cache=True)
def _nbr(a, L, d):
    x = a // L; y = a % L
    if d == 0: return x * L + (y + 1) % L
    if d == 1: return ((x + 1) % L) * L + y
    if d == 2: return x * L + (y - 1) % L
    return ((x - 1) % L) * L + y

@njit(cache=True)
def _ends(b, L):
    a = b // 2
    return a, _nbr(a, L, b % 2)

@njit(cache=True)
def _add(lst, pos, cnt, c, e):
    if pos[c, e] < 0:
        lst[c, cnt[c]] = e; pos[c, e] = cnt[c]; cnt[c] += 1

@njit(cache=True)
def _rem(lst, pos, cnt, c, e):
    k = pos[c, e]
    if k >= 0:
        last = lst[c, cnt[c] - 1]; lst[c, k] = last; pos[c, last] = k
        cnt[c] -= 1; pos[c, e] = -1

@njit(cache=True)
def _upd_site(a, s, lst, pos, cnt, L):
    if s[a] == 0: _add(lst, pos, cnt, 0, a)
    else: _rem(lst, pos, cnt, 0, a)
    for d in range(4):                       # the 4 bonds touching a
        if d == 0: b = 2 * a
        elif d == 1: b = 2 * a + 1
        elif d == 2: b = 2 * _nbr(a, L, 2)
        else: b = 2 * _nbr(a, L, 3) + 1
        i, j = _ends(b, L)
        if s[i] == 0 and s[j] == 0: _add(lst, pos, cnt, 1, b)
        else: _rem(lst, pos, cnt, 1, b)
        if (s[i] == 1 and s[j] == 2) or (s[i] == 2 and s[j] == 1): _add(lst, pos, cnt, 2, b)
        else: _rem(lst, pos, cnt, 2, b)

@njit(cache=True)
def zgb_kmc(L, y, kr, t_warm, t_meas, seed):
    np.random.seed(seed)
    N = L * L; NB = 2 * N
    s = np.zeros(N, np.int8)
    lst = np.zeros((3, NB), np.int64); pos = -np.ones((3, NB), np.int64); cnt = np.zeros(3, np.int64)
    for a in range(N): _add(lst, pos, cnt, 0, a)
    for b in range(NB): _add(lst, pos, cnt, 1, b)
    kco = y; ko2 = (1 - y) / 2
    t = 0.0; nCO = 0; nO = 0; sCO = 0.0; sO = 0.0; nprod = 0; nev = 0
    tend = t_warm + t_meas
    while t < tend:
        R0 = kco * cnt[0]; R1 = ko2 * cnt[1]; R2 = kr * cnt[2]; R = R0 + R1 + R2
        if R <= 0.0:                                   # absorbing state: nothing can happen any more
            if t < t_warm: t = t_warm
            sCO += nCO * (tend - t); sO += nO * (tend - t); t = tend
            break
        tau = -np.log(np.random.random()) / R
        if t + tau > t_warm:                           # time-weighted coverages in the window
            lo = max(t, t_warm); hi = min(t + tau, tend)
            sCO += nCO * (hi - lo); sO += nO * (hi - lo)
        t += tau
        if t >= tend: break
        u = np.random.random() * R; nev += 1
        if u < R0:
            a = lst[0, np.random.randint(cnt[0])]
            s[a] = 1; nCO += 1; _upd_site(a, s, lst, pos, cnt, L)
        elif u < R0 + R1:
            b = lst[1, np.random.randint(cnt[1])]; i, j = _ends(b, L)
            s[i] = 2; s[j] = 2; nO += 2
            _upd_site(i, s, lst, pos, cnt, L); _upd_site(j, s, lst, pos, cnt, L)
        else:
            b = lst[2, np.random.randint(cnt[2])]; i, j = _ends(b, L)
            s[i] = 0; s[j] = 0; nCO -= 1; nO -= 1
            if t > t_warm: nprod += 1
            _upd_site(i, s, lst, pos, cnt, L); _upd_site(j, s, lst, pos, cnt, L)
    return sCO / t_meas / N, sO / t_meas / N, nprod / t_meas / N, nev

t0 = time.time()
L = 64
print("=== (a) k_r = 1e3 vs original ZGB, L = 64, 1500 + 1500 (MCS or time units), seeds 1-3 ===")
for y in (0.45, 0.50):
    a = np.array([zgb(L, y, 1500, 1500, sd)[:3] for sd in (1, 2, 3)])
    b = np.array([zgb_kmc(L, y, 1e3, 1500.0, 1500.0, sd)[:3] for sd in (1, 2, 3)])
    for name, arr in (("ZGB (rejection)", a), ("KMC k_r=1e3   ", b)):
        m = arr.mean(0); e = arr.std(0, ddof=1) / np.sqrt(3)
        print(f"  y={y:.2f} {name}: theta_O = {m[1]:.4f} +/- {e[1]:.4f}, theta_CO = {m[0]:.4f} +/- {e[0]:.4f}, R_CO2 = {m[2]:.4f} +/- {e[2]:.4f}")
print(f"  ({time.time()-t0:.0f} s)")
print("=== (b) reactive window vs k_r (L = 64, 1500 + 1500 time units, seed 1) ===")
ys = np.round(np.arange(0.20, 0.601, 0.02), 2)
for kr in (1e3, 10.0, 1.0):
    rows = np.array([(y, *zgb_kmc(L, y, kr, 1500.0, 1500.0, 1)[:3]) for y in ys])
    act = rows[rows[:, 3] > 1e-3]
    print(f"  k_r = {kr:g}: window {act[0,0]:.2f} <= y <= {act[-1,0]:.2f}; max R_CO2 = {rows[:,3].max():.4f} at y = {rows[np.argmax(rows[:,3]),0]:.2f}")
    print("     y   : " + " ".join(f"{r[0]:.2f}" for r in rows))
    print("    thCO : " + " ".join(f"{r[1]:.2f}" for r in rows))
    print("    thO  : " + " ".join(f"{r[2]:.2f}" for r in rows))
    print("    R    : " + " ".join(f"{r[3]:.3f}" for r in rows))
print(f"  ({time.time()-t0:.0f} s)")
# fine scan near the upper edge for k_r = 1e3 and 1 (seed 1)
for kr in (1e3, 1.0):
    yy = np.round(np.arange(0.48, 0.561, 0.005), 3) if kr > 1 else np.round(np.arange(0.36, 0.461, 0.005), 3)
    rows = np.array([(y, *zgb_kmc(L, y, kr, 1500.0, 1500.0, 1)[:3]) for y in yy])
    act = rows[rows[:, 3] > 1e-3]
    print(f"  fine scan k_r={kr:g}: last reactive y = {act[-1,0]:.3f} (R = {act[-1,3]:.4f}, theta_CO = {act[-1,1]:.3f})")
# (c) physical time: Hertz-Knudsen impingement rate per site
kB = 1.380649e-23; amu = 1.66053907e-27
p, T, m, Asite = 1e-4, 500.0, 28.0 * amu, 1.0 / 1.5e19     # 1e-4 Pa (1e-6 mbar) CO, 500 K, 1.5e15 sites/cm^2
F = p / np.sqrt(2 * np.pi * m * kB * T) * Asite
print(f"(c) CO impingement per site at p=1e-4 Pa, T=500 K: {F:.3e} 1/s -> 1 MCS (y=1 scale) ~ {1/F:.3e} s")
print(f"total {time.time()-t0:.0f} s")
