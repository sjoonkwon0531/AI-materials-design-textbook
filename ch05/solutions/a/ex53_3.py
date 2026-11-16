"""Ex 5.3.3: detailed balance of lattice-gas hop rules, checked against Kawasaki Metropolis MC.
Rules: 0 = bond counting at origin  k = nu exp(-b(Ea + n Eb));
       1 = arrival rule              k = nu exp(-b(Ea - n' Eb)), n' = occupied nbrs of target (origin excluded);
       2 = saturated                 k = nu exp(-b(Ea + min(n,1) Eb)).
Square lattice L=40, N=160 (theta=0.1), Ea=0.5 eV, Eb=0.1 eV, nu=1e13 1/s.
Equilibrium observable: mean number of occupied neighbours per atom <z> (time-weighted in KMC)."""
import numpy as np
from numba import njit
KB = 8.617333262e-5

@njit(cache=True)
def kmc(L, N, Ea, Eb, nu, T, rule, n_equil, n_prod, seed):
    np.random.seed(seed)
    beta = 1.0 / (KB * T)
    occ = -np.ones((L, L), np.int64); pos = np.zeros((N, 2), np.int64); unw = np.zeros((N, 2))
    p = 0
    while p < N:
        i = np.random.randint(L); j = np.random.randint(L)
        if occ[i, j] < 0:
            occ[i, j] = p; pos[p, 0] = i; pos[p, 1] = j; p += 1
    di = np.array([1, -1, 0, 0]); dj = np.array([0, 0, 1, -1])
    rates = np.zeros(4 * N)
    t = 0.0; tw = 0.0; zsum = 0.0
    for ev in range(n_equil + n_prod):
        if ev == n_equil:
            t = 0.0; unw[:, :] = 0.0
        R = 0.0; ztot = 0
        for a in range(N):
            i = pos[a, 0]; j = pos[a, 1]
            n = 0
            for d in range(4):
                if occ[(i + di[d]) % L, (j + dj[d]) % L] >= 0:
                    n += 1
            ztot += n
            for d in range(4):
                ti = (i + di[d]) % L; tj = (j + dj[d]) % L
                if occ[ti, tj] >= 0:
                    rates[4 * a + d] = 0.0
                    continue
                if rule == 0:
                    k = nu * np.exp(-beta * (Ea + n * Eb))
                elif rule == 1:
                    n2 = 0
                    for d2 in range(4):
                        ui = (ti + di[d2]) % L; uj = (tj + dj[d2]) % L
                        if (ui != i or uj != j) and occ[ui, uj] >= 0:
                            n2 += 1
                    k = nu * np.exp(-beta * (Ea - n2 * Eb))
                else:
                    k = nu * np.exp(-beta * (Ea + min(n, 1) * Eb))
                rates[4 * a + d] = k; R += k
        tau = -np.log(np.random.random()) / R
        if ev >= n_equil:
            zsum += tau * ztot / N; tw += tau
        r = np.random.random() * R; c = 0.0; e = 0
        while True:
            c += rates[e]
            if c >= r:
                break
            e += 1
        a = e // 4; d = e % 4
        i = pos[a, 0]; j = pos[a, 1]; ni = (i + di[d]) % L; nj = (j + dj[d]) % L
        occ[i, j] = -1; occ[ni, nj] = a; pos[a, 0] = ni; pos[a, 1] = nj
        unw[a, 0] += di[d]; unw[a, 1] += dj[d]
        t += tau
    msd = (unw ** 2).sum() / N
    return zsum / tw, msd / (4.0 * t), t

@njit(cache=True)
def kawasaki(L, N, Eb, T, nsweep_eq, nsweep, seed):
    np.random.seed(seed)
    beta = 1.0 / (KB * T)
    occ = -np.ones((L, L), np.int64); pos = np.zeros((N, 2), np.int64)
    p = 0
    while p < N:
        i = np.random.randint(L); j = np.random.randint(L)
        if occ[i, j] < 0:
            occ[i, j] = p; pos[p, 0] = i; pos[p, 1] = j; p += 1
    di = np.array([1, -1, 0, 0]); dj = np.array([0, 0, 1, -1])
    zs = 0.0; cnt = 0
    for sw in range(nsweep_eq + nsweep):
        for _ in range(N):
            a = np.random.randint(N); d = np.random.randint(4)
            i = pos[a, 0]; j = pos[a, 1]; ti = (i + di[d]) % L; tj = (j + dj[d]) % L
            if occ[ti, tj] >= 0:
                continue
            n = 0; n2 = 0
            for d2 in range(4):
                if occ[(i + di[d2]) % L, (j + dj[d2]) % L] >= 0:
                    n += 1
                ui = (ti + di[d2]) % L; uj = (tj + dj[d2]) % L
                if (ui != i or uj != j) and occ[ui, uj] >= 0:
                    n2 += 1
            dE = Eb * (n - n2)                      # E = -Eb * (#bonds)
            if dE <= 0 or np.random.random() < np.exp(-beta * dE):
                occ[i, j] = -1; occ[ti, tj] = a; pos[a, 0] = ti; pos[a, 1] = tj
        if sw >= nsweep_eq:
            z = 0
            for a in range(N):
                for d2 in range(4):
                    if occ[(pos[a, 0] + di[d2]) % L, (pos[a, 1] + dj[d2]) % L] >= 0:
                        z += 1
            zs += z / N; cnt += 1
    return zs / cnt

L, N, Ea, Eb, nu = 40, 160, 0.5, 0.1, 1e13
T = 800.0
print(f"=== equilibrium check at T={T:.0f} K (kT={KB*T:.4f} eV, beta Eb={Eb/(KB*T):.3f}) ===")
zs = [kawasaki(L, N, Eb, T, 2000, 40000, 10 + s) for s in range(3)]
print(f"  Kawasaki Metropolis MC (Boltzmann reference), 3 seeds: <z> = {np.mean(zs):.4f} +/- {np.std(zs, ddof=1)/np.sqrt(3):.4f}")
Ds = {}
for rule, name in [(0, "bond counting (origin)"), (1, "arrival rule"), (2, "saturated min(n,1)")]:
    out = [kmc(L, N, Ea, Eb, nu, T, rule, 200000, 1500000, 20 + s) for s in range(3)]
    z = np.array([o[0] for o in out]); D = np.array([o[1] for o in out])
    Ds[rule] = D.mean()
    print(f"  KMC {name:24s}: <z> = {z.mean():.4f} +/- {z.std(ddof=1)/np.sqrt(3):.4f},  D/D_single = {D.mean()/(nu*np.exp(-Ea/(KB*T))):.3f} +/- {D.std(ddof=1)/np.sqrt(3)/(nu*np.exp(-Ea/(KB*T))):.3f}")
z0 = [kmc(L, N, Ea, 0.0, nu, T, 0, 100000, 300000, 30 + s)[0] for s in range(2)]
print(f"  (no interaction, Eb=0: <z> = {np.mean(z0):.4f}; random placement value 4*(N-1)/(L^2-1) = {4*(N-1)/(L*L-1):.4f})")
print("=== saturated rule in the Lab 5.5 Part A setting (Eb=0.1, 1e5+3e5 events, seed 100) ===")
for rule in (0, 2):
    Ts = np.array([300.0, 350.0, 400.0, 500.0, 600.0, 800.0]); Dl = []; zl = []
    for TT in Ts:
        z, D, _ = kmc(L, N, Ea, Eb, nu, TT, rule, 100000, 300000, 100)
        Dl.append(D); zl.append(z)
    Dl = np.array(Dl)
    sl = np.polyfit(1 / (KB * Ts), np.log(Dl), 1)[0]
    print(f"  rule {rule}: D/D_single = " + " ".join(f"{d/(nu*np.exp(-Ea/(KB*TT))):.3f}" for d, TT in zip(Dl, Ts))
          + f";  <z> = " + " ".join(f"{z:.2f}" for z in zl) + f";  apparent Ea = {-sl:.3f} eV")
