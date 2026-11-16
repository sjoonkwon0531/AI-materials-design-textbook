"""Ex 8.6.3: GP-EI noise variance on the OOD-noisy landscapes (sigma_y = 0.05), same landscapes/seeds as Lab 8.16."""
import time, numpy as np
from common import lab
M = lab("86_dro"); t0 = time.time()
rng = np.random.default_rng(99)
G = M.sample_gmm(150, rng); Fs = M.global_max(G); x0 = M.x0_batch(150, rng)
print(f"landscapes ready ({time.time()-t0:.0f} s)", flush=True)
sds = []
for nv in (1e-4, 1e-3, 1e-2, 1e-1, 3e-1):
    best_true, rec_true, R10 = [], [], []
    for i in range(150):
        nrng = np.random.default_rng(1000 + i); seen = []
        true = lambda x, i=i: float(M.f_np(G, np.asarray(x, float)[None], idx=slice(i, i + 1))[0])
        def fun(x):
            v = true(x); seen.append(v); return v + 0.05 * nrng.standard_normal()
        Yobs = M.gp_ei(fun, x0[i], np.random.default_rng(3000 + i), noise=nv)
        seen = np.array(seen); Yobs = np.array(Yobs)
        if nv == 1e-4: sds.append(Yobs.std())
        best_true.append(Fs[i] - seen.max()); rec_true.append(Fs[i] - seen[int(np.argmax(Yobs))])
        R10.append(Fs[i] - seen[:11].max())
    b, r = np.array(best_true), np.array(rec_true)
    print(f"noise var {nv:7.0e}: regret@50 (best true value queried) {b.mean():.3f} +/- {b.std(ddof=1)/np.sqrt(150):.3f} | "
          f"P(<0.05) {np.mean(b<0.05):.2f} | regret of the point with best OBSERVED y {r.mean():.3f} +/- {r.std(ddof=1)/np.sqrt(150):.3f} | "
          f"regret@10 {np.mean(R10):.3f} | {time.time()-t0:.0f} s", flush=True)
print(f"sd of observed y within a run (noise var 1e-4 runs): median {np.median(sds):.3f}; implied standardized noise var (0.05/sd)^2 = {np.median((0.05/np.array(sds))**2):.3f}")
