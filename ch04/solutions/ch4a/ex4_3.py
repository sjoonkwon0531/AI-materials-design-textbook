"""Verification code for Ch.4 exercises 4.3.1 (O(N^3) timing) and 4.3.5 (nearsightedness)."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import time
import numpy as np
import scipy.linalg as la

# ---------------- 4.3.1: Lab 4.3 timing (same code) ----------------
np.random.seed(0)
sizes = [200, 400, 800, 1200, 1600, 2400, 3200]
print("=== 4.3.1 Lab 4.3 re-run (single thread), 3 repetitions ===")
mats = {}
for N in sizes:
    A = np.random.randn(N, N); mats[N] = 0.5*(A + A.T)
tails = []
for rep in range(3):
    times = []
    for N in sizes:
        reps = 5 if N <= 800 else (3 if N <= 1600 else 1)
        t0 = time.perf_counter()
        for _ in range(reps):
            la.eigvalsh(mats[N])
        times.append((time.perf_counter() - t0)/reps)
    s_ = np.array(sizes, float); t = np.array(times)
    p_full = np.polyfit(np.log(s_), np.log(t), 1)[0]
    p_tail = np.polyfit(np.log(s_[s_ >= 800]), np.log(t[s_ >= 800]), 1)[0]
    tails.append(p_tail)
    print(f"  rep {rep+1}: " + " ".join(f"N={N}:{tt:.4f}s" for N, tt in zip(sizes, times))
          + f"  | full p={p_full:.2f}, tail p={p_tail:.2f}")
p_tail = float(np.median(tails))
print(f"  median tail exponent p={p_tail:.2f}")
for p in [3.0, p_tail]:
    t500 = 10.0*10**p
    print(f"  50 atoms 10 s -> 500 atoms with p={p:.2f}: {t500:.0f} s = {t500/3600:.2f} h")
# MD consequence: 10^4 steps (10 ps at 1 fs) for each
for natom, tstep in [(50, 10.0), (500, 10.0*1000)]:
    tot = tstep*1e4
    print(f"  {natom} atoms, 1e4 MD steps: {tot/3600:.1f} h = {tot/86400:.1f} days")

# ---------------- 4.3.5: density-matrix decay, insulator vs metal ----------------
print("\n=== 4.3.5 density matrix decay (1D tight-binding ring, half filling) ===")


def density_matrix(Nsite, t1, t2, onsite=0.0):
    """Ring with alternating hoppings t1,t2 and alternating onsite +/-onsite; P = sum_occ |phi><phi|."""
    H = np.zeros((Nsite, Nsite))
    for i in range(Nsite):
        j = (i + 1) % Nsite
        t = t1 if i % 2 == 0 else t2
        H[i, j] = H[j, i] = -t
        H[i, i] = onsite*(1 if i % 2 == 0 else -1)
    e, v = la.eigh(H)
    nocc = Nsite//2
    P = v[:, :nocc] @ v[:, :nocc].T
    gap = e[nocc] - e[nocc - 1]
    return P, gap


Nsite = 402          # Nsite/2 odd -> closed shell for the uniform (metallic) ring
cases = {"metal  (t1=t2=1)": (1.0, 1.0, 0.0),
         "insul. (t1=1,t2=0.8)": (1.0, 0.8, 0.0),
         "insul. (t1=1,t2=0.5)": (1.0, 0.5, 0.0),
         "insul. (onsite +/-0.5)": (1.0, 1.0, 0.5)}
dists = [1, 5, 11, 21, 41, 81]
for name, (t1, t2, d) in cases.items():
    P, gap = density_matrix(Nsite, t1, t2, d)
    row = np.abs(P[0])
    # odd distances only (for the bipartite chain at half filling, P_0j = 0 for even j != 0)
    vals = [row[j] for j in dists]
    jj = np.arange(1, 120, 2); yy = row[jj]
    msk = yy > 1e-13
    slope_exp = np.polyfit(jj[msk], np.log(yy[msk]), 1)[0]
    slope_pow = np.polyfit(np.log(jj[msk]), np.log(yy[msk]), 1)[0]
    print(f"  {name:24s} gap={gap:.3f}  |P_0j| at j={dists}: " + " ".join(f"{v:.1e}" for v in vals))
    print(f"  {'':24s} fit exp: decay length xi={-1/slope_exp:7.2f} sites | fit power law exponent={slope_pow:.2f}")

# truncation: keep P_ij only for |i-j|<=R, measure loss of idempotency
print("\n  truncation: drop P_ij with |i-j|>R and measure loss of idempotency (P^2=P):")


def trunc_error(t1, t2, d, R):
    H = np.zeros((Nsite, Nsite))
    for i in range(Nsite):
        j = (i + 1) % Nsite
        H[i, j] = H[j, i] = -(t1 if i % 2 == 0 else t2)
        H[i, i] = d*(1 if i % 2 == 0 else -1)
    P, _ = density_matrix(Nsite, t1, t2, d)
    idx = np.arange(Nsite)
    dist = np.abs(idx[:, None] - idx[None, :]); dist = np.minimum(dist, Nsite - dist)
    Pt = np.where(dist <= R, P, 0.0)
    # quantity sensitive to the whole matrix: idempotency error ||Pt^2-Pt|| / ||P||
    return np.linalg.norm(Pt@Pt - Pt)/np.linalg.norm(P)


for name, (t1, t2, d) in cases.items():
    print(f"  {name:24s} idempotency error ||P_R^2-P_R||/||P|| for R=5,11,21,41: " +
          " ".join(f"{trunc_error(t1, t2, d, R):.1e}" for R in [5, 11, 21, 41]))
