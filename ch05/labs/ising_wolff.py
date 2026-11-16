"""Lab 5.7B - Metropolis (checkerboard, Numba) vs Wolff cluster at T_c:
integrated autocorrelation time and wall time per independent sample.
Run: python ising_wolff.py   (about 3-4 min on a shared 2-core machine)"""
import math, time, json, os
import numpy as np
from numba import njit

TC = 2.0 / math.log(1.0 + math.sqrt(2.0))
OUT = os.path.dirname(os.path.abspath(__file__))

@njit(cache=True)
def seed(sd):
    np.random.seed(sd)

@njit(cache=True)
def metro_series(s, beta, nsweep):
    """checkerboard Metropolis; returns |m| after every sweep"""
    L = s.shape[0]; N = L * L
    w4 = math.exp(-4 * beta); w8 = math.exp(-8 * beta)
    out = np.empty(nsweep)
    M = 0
    for i in range(L):
        for j in range(L):
            M += s[i, j]
    for n in range(nsweep):
        for c in range(2):
            for i in range(L):
                ip = i + 1 if i + 1 < L else 0; im = i - 1 if i > 0 else L - 1
                for j in range((i + c) % 2, L, 2):
                    jp = j + 1 if j + 1 < L else 0; jm = j - 1 if j > 0 else L - 1
                    x = s[i, j] * (s[ip, j] + s[im, j] + s[i, jp] + s[i, jm])
                    flip = x <= 0
                    if not flip:
                        flip = np.random.random() < (w4 if x == 2 else w8)
                    if flip:
                        M -= 2 * s[i, j]; s[i, j] = -s[i, j]
        out[n] = abs(M) / N
    return out

@njit(cache=True)
def wolff_series(s, beta, nstep):
    """Wolff single-cluster updates; returns |m| and cluster size after every update"""
    L = s.shape[0]; N = L * L
    padd = 1.0 - math.exp(-2.0 * beta)          # bond activation probability
    stack = np.empty(N, np.int64)
    mser = np.empty(nstep); cser = np.empty(nstep, np.int64)
    M = 0
    for i in range(L):
        for j in range(L):
            M += s[i, j]
    for n in range(nstep):
        k = np.random.randint(N); i0 = k // L; j0 = k % L
        s0 = s[i0, j0]
        s[i0, j0] = -s0                           # flip on push => each site enters once
        stack[0] = k; top = 1; size = 1
        while top > 0:
            top -= 1; k = stack[top]; i = k // L; j = k % L
            for d in range(4):
                if d == 0:   ii = (i + 1) % L; jj = j
                elif d == 1: ii = (i - 1) % L; jj = j
                elif d == 2: ii = i; jj = (j + 1) % L
                else:        ii = i; jj = (j - 1) % L
                if s[ii, jj] == s0 and np.random.random() < padd:
                    s[ii, jj] = -s0
                    stack[top] = ii * L + jj; top += 1; size += 1
        M -= 2 * s0 * size
        mser[n] = abs(M) / N; cser[n] = size
    return mser, cser

def tau_int(x, c=6.0):
    """Sokal automatic windowing: smallest W with W >= c * tau(W)"""
    x = np.asarray(x) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]
    rho = acf / acf[0]
    tau = 0.5
    for W in range(1, n):
        tau += rho[W]
        if W >= c * tau:
            break
    err = tau * math.sqrt(2 * (2 * W + 1) / n)
    return tau, err

if __name__ == "__main__":
    beta = 1 / TC
    seed(11); s = np.ones((8, 8), np.int64); metro_series(s, beta, 10); wolff_series(s, beta, 10)
    rows = []
    print("=== T = T_c = 2.2692, observable |m|; tau in sweeps; t_ind = 2 tau x (wall time per sweep) ===")
    print("   L | Metropolis: tau_int     t_ind[ms] |  Wolff: <C>/N   tau_int     t_ind[ms]")
    for L, nsw in [(8, 200000), (16, 200000), (32, 200000), (64, 200000), (128, 120000)]:
        N = L * L
        seed(100 + L); s = np.ones((L, L), np.int64)
        metro_series(s, beta, 5000)                                   # equilibrate
        t0 = time.perf_counter(); mm = metro_series(s, beta, nsw); tm = (time.perf_counter() - t0) / nsw
        tauM, eM = tau_int(mm)
        # Wolff: run long enough to cover ~ 4e4 sweep-equivalents (plenty: tau ~ 1)
        seed(200 + L); s = np.ones((L, L), np.int64)
        wolff_series(s, beta, 2000)
        nst = 200000
        t0 = time.perf_counter(); mw, cw = wolff_series(s, beta, nst); tw_step = (time.perf_counter() - t0) / nst
        cfrac = cw.mean() / N                          # sweep-equivalents per cluster update
        tauW_steps, eW = tau_int(mw)
        tauW = tauW_steps * cfrac; eWs = eW * cfrac   # convert to sweeps
        tw_sweep = tw_step / cfrac
        rows.append(dict(L=L, tauM=tauM, eM=eM, tM=tm, tauW=tauW, eW=eWs, tW=tw_sweep, cfrac=cfrac,
                         tindM=2 * tauM * tm, tindW=2 * tauW * tw_sweep))
        print(f" {L:3d} |        {tauM:8.1f}+/-{eM:6.1f}  {2*tauM*tm*1e3:9.3f} |  {cfrac:6.3f}   "
              f"{tauW:6.2f}+/-{eWs:4.2f}  {2*tauW*tw_sweep*1e3:9.4f}")
    Ls = np.array([r["L"] for r in rows], float)
    zM = np.polyfit(np.log(Ls[1:]), np.log([r["tauM"] for r in rows][1:]), 1)[0]
    zW = np.polyfit(np.log(Ls[1:]), np.log([r["tauW"] for r in rows][1:]), 1)[0]
    print(f"fit L=16..128: tau_Metropolis ~ L^{zM:.2f},  tau_Wolff(sweeps) ~ L^{zW:.2f}")
    r = rows[-1]
    print(f"L=128: Wolff delivers an independent sample {r['tindM']/r['tindW']:.0f}x faster in wall time")
    json.dump(dict(rows=rows, zM=zM, zW=zW), open(os.path.join(OUT, "ising_wolff.json"), "w"))
