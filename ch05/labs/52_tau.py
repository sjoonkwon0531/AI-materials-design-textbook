"""Lab 5.2 (key 52_tau): autocorrelation, error bars and critical slowing down.
2D Ising (J = 1) at the exact critical point beta_c = ln(1+sqrt 2)/2.
Single-spin Metropolis (random site, 1 sweep = L^2 trials) vs Wolff cluster algorithm.
Time unit for both: one sweep (Wolff: cluster flips x <C>/L^2).
Runtime about 3-4 min on 2 cores (single-threaded). Seeds fixed; tau values vary by ~5-25 % with seed.
"""
import time
import numpy as np
from isingkern import run_metro2d, run_wolff2d, tau_int, autocorr, blocking

bc = 0.5 * np.log(1 + np.sqrt(2))
t0 = time.time()

metro_runs = {8: 600000, 16: 400000, 32: 500000, 64: 400000}
wolff_runs = {8: 200000, 16: 200000, 32: 150000, 64: 80000, 128: 40000}

print("=== Metropolis (random-site), T = T_c, 2000 eq sweeps; tau in sweeps ===")
print("   L   sweeps    tau_int(m^2)        tau_int(E)       CPU s")
res_m = {}
for L, n in metro_runs.items():
    t = time.time()
    E, M = run_metro2d(L, bc, 2000, n, 100 + L)
    m2 = (M / L**2) ** 2
    tm = tau_int(m2)
    te = tau_int(E)
    res_m[L] = dict(tm=tm[0], tm_err=tm[1], te=te[0], te_err=te[1], cpu=time.time() - t, n=n)
    if L == 32:
        rho_m32 = autocorr(m2, 1500)
        blk = blocking(m2)
        m2_32 = m2
    print(f"  {L:3d}  {n:7d}   {tm[0]:7.1f} +/- {tm[1]:5.1f}   {te[0]:6.1f} +/- {te[1]:4.1f}   {time.time()-t:6.1f}")

print("=== Wolff, T = T_c, 1000 eq clusters; tau converted to sweeps = tau_cluster * <C>/L^2 ===")
print("   L  clusters  <C>/L^2   tau_int(m^2)      tau_int(E)      CPU s")
res_w = {}
for L, n in wolff_runs.items():
    t = time.time()
    E, M, C = run_wolff2d(L, bc, 1000, n, 200 + L)
    f = C.mean() / L**2
    tm = tau_int((M / L**2) ** 2)
    te = tau_int(E)
    res_w[L] = dict(f=f, C=C.mean(), tm=tm[0] * f, tm_err=tm[1] * f, te=te[0] * f, te_err=te[1] * f,
                    cpu=time.time() - t, n=n)
    if L == 32:
        rho_w32 = autocorr((M / L**2) ** 2, 400)
        f32 = f
    print(f"  {L:3d}  {n:7d}   {f:.4f}   {tm[0]*f:6.3f} +/- {tm[1]*f:5.3f}   {te[0]*f:6.3f} +/- {te[1]*f:5.3f}  {time.time()-t:6.1f}")

# dynamic exponents from log-log fits
Lm = np.array(list(metro_runs))
Lw = np.array(list(wolff_runs))
zm = np.polyfit(np.log(Lm), np.log([res_m[L]["tm"] for L in Lm]), 1)[0]
zm2 = np.polyfit(np.log(Lm[1:]), np.log([res_m[L]["tm"] for L in Lm[1:]]), 1)[0]
zw_m = np.polyfit(np.log(Lw), np.log([res_w[L]["tm"] for L in Lw]), 1)[0]
zw_e = np.polyfit(np.log(Lw), np.log([res_w[L]["te"] for L in Lw]), 1)[0]
gC = np.polyfit(np.log(Lw), np.log([res_w[L]["C"] for L in Lw]), 1)[0]
print("=== fits ===")
print(f"  Metropolis z (m^2): L=8-64 -> {zm:.2f};  L=16-64 -> {zm2:.2f}")
print(f"  Wolff z (m^2) = {zw_m:.2f},  Wolff z (E) = {zw_e:.2f}   (L = 8-128)")
print(f"  Wolff <C> ~ L^{gC:.3f}   (gamma/nu = 1.75)")

# blocking demo on Metropolis L = 32
naive = blk[0]
plateau = blk[np.argmax(blk[:-3])] if len(blk) > 4 else blk[-1]
print("=== blocking analysis, Metropolis L=32, observable m^2 ===")
print("  level  blocks  std.err")
for k, e in enumerate(blk):
    if k % 2 == 0 or k == len(blk) - 1:
        print(f"  {k:5d}  {res_m[32]['n'] // 2**k:6d}  {e:.2e}")
print(f"  <m^2> = {m2_32.mean():.4f};  naive err = {naive:.2e};  plateau err = {plateau:.2e}")
print(f"  (plateau/naive)^2 = {(plateau/naive)**2:.0f}   vs   2*tau_int = {2*res_m[32]['tm']:.0f}")

# cost of one independent sample at L = 64
for L in [64]:
    cm = res_m[L]["cpu"] / res_m[L]["n"] * 2 * res_m[L]["tm"]
    cw = res_w[L]["cpu"] / (res_w[L]["n"] * res_w[L]["f"]) * 2 * res_w[L]["tm"]
    print(f"=== CPU seconds per independent sample (2 tau_int) at L={L}: Metropolis {cm*1e3:.1f} ms, Wolff {cw*1e3:.2f} ms, ratio {cm/cw:.0f} ===")
print(f"total {time.time()-t0:.0f} s")

np.savez("../ch05/labs/52_tau_results.npz",
         Lm=Lm, tm=[res_m[L]["tm"] for L in Lm], tm_err=[res_m[L]["tm_err"] for L in Lm],
         te_m=[res_m[L]["te"] for L in Lm],
         Lw=Lw, tw=[res_w[L]["tm"] for L in Lw], tw_err=[res_w[L]["tm_err"] for L in Lw],
         te_w=[res_w[L]["te"] for L in Lw],
         rho_m32=rho_m32, rho_w32=rho_w32, f32=f32, blk=blk, zm=zm, zw_m=zw_m, zw_e=zw_e)
