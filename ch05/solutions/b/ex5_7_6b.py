"""Ex 5.7.6 (Numba counterpart): batch of B independent lattices (one temperature each) swept with
numba.prange over the batch index; checkerboard + acceptance table as in Lab 5.10. L=64, 200 sweeps, best of 5."""
import math, time, numpy as np, numba
from numba import njit, prange
@njit(cache=True)
def cb_sweeps(s, beta, nsweep):
    L = s.shape[0]; w4 = math.exp(-4*beta); w8 = math.exp(-8*beta)
    for _ in range(nsweep):
        for c in range(2):
            for i in range(L):
                ip = i+1 if i+1 < L else 0; im = i-1 if i > 0 else L-1
                for j in range((i+c) % 2, L, 2):
                    jp = j+1 if j+1 < L else 0; jm = j-1 if j > 0 else L-1
                    x = s[i, j]*(s[ip, j]+s[im, j]+s[i, jp]+s[i, jm])
                    if x <= 0: s[i, j] = -s[i, j]
                    elif x == 2:
                        if np.random.random() < w4: s[i, j] = -s[i, j]
                    elif np.random.random() < w8: s[i, j] = -s[i, j]
@njit(parallel=True, cache=True)
def batch(S, betas, nsweep):
    for b in prange(S.shape[0]):              # each lattice/temperature is an independent task
        cb_sweeps(S[b], betas[b], nsweep)
@njit(cache=True)
def batch_serial(S, betas, nsweep):
    for b in range(S.shape[0]):
        cb_sweeps(S[b], betas[b], nsweep)
L = 64; NSW = 200
print(f"numba {numba.__version__}, threads {numba.get_num_threads()}")
for B in (1, 2, 4, 8, 16):
    S = np.ones((B, L, L), np.int64); betas = 1/np.linspace(1.6, 3.0, B) if B > 1 else np.array([1/2.4])
    batch(S, betas, 1); batch_serial(S, betas, 1)
    tp = ts = 1e30
    for _ in range(5):
        t0 = time.perf_counter(); batch(S, betas, NSW); tp = min(tp, time.perf_counter()-t0)
        t0 = time.perf_counter(); batch_serial(S, betas, NSW); ts = min(ts, time.perf_counter()-t0)
    n = B*NSW*L*L
    print(f"B={B:2d}: prange {tp/n*1e9:5.1f} ns/update ({n/tp/1e6:6.1f} Mupd/s), serial {ts/n*1e9:5.1f} ns/update, parallel speed-up x{ts/tp:.2f}")
