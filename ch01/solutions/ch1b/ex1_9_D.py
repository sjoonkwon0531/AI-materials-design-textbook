"""Ex 1.9.D: finite-size scaling of 2D Ising susceptibility, L = 8,16,32,64 (Wolff cluster MC).
gamma/nu from chi'_max(L) and chi(Tc,L) ~ L^{gamma/nu};
1/nu from dU4/dT at Tc ~ L^{1/nu} and from T*(L)-Tc ~ L^{-1/nu}."""
import numpy as np, time
from ising_lib import seed, run_wolff2d
seed(11); rng = np.random.default_rng(11)
Tc = 2/np.log(1+np.sqrt(2))
Ls = [8, 16, 32, 64]; Ts = np.round(np.arange(2.20, 2.4001, 0.01), 3)
n_meas = {8: 200000, 16: 200000, 32: 200000, 64: 200000}
res = {}
t0 = time.time()
for L in Ls:
    s = np.ones((L, L), dtype=np.int64); rows = []
    for T in Ts:
        m = run_wolff2d(s, 1/T, 2000, n_meas[L])
        a = np.abs(m); m2 = np.mean(m**2); m4 = np.mean(m**4)
        chi_p = (1/T)*L*L*(m2 - a.mean()**2); chi = (1/T)*L*L*m2; U = 1 - m4/(3*m2**2)
        rows.append((T, chi_p, chi, U))
    res[L] = np.array(rows); print(f"L={L} done ({time.time()-t0:.0f}s)", flush=True)
np.save("ex1_9_D_res.npy", np.array([res[L] for L in Ls]))
# quadratic fit around the chi' peak for T*(L) and chi'_max
Tstar, chimax, chiTc, slope = [], [], [], []
for L in Ls:
    r = res[L]; i = np.argmax(r[:, 1]); sl = slice(max(i-3, 0), min(i+4, len(Ts)))
    c = np.polyfit(r[sl, 0], r[sl, 1], 2); Tp = -c[1]/(2*c[0]); Tstar.append(Tp); chimax.append(np.polyval(c, Tp))
    w = np.abs(r[:, 0]-Tc) < 0.035
    chiTc.append(np.exp(np.polyval(np.polyfit(r[w, 0], np.log(r[w, 2]), 2), Tc)))
    slope.append(-np.polyfit(r[w, 0], r[w, 3], 2) @ np.array([2*Tc, 1, 0]))   # -dU/dT at Tc
    print(f"L={L:3d}: T*={Tp:.4f}, chi'_max={chimax[-1]:.3f}, chi(Tc)={chiTc[-1]:.3f}, U4(Tc)~{np.interp(Tc, r[:,0], r[:,3]):.4f}, -dU/dT|Tc={slope[-1]:.3f}")
lL = np.log(Ls)
g1 = np.polyfit(lL, np.log(chimax), 1)[0]; g2 = np.polyfit(lL, np.log(chiTc), 1)[0]
inu1 = np.polyfit(lL, np.log(slope), 1)[0]
inu2 = -np.polyfit(lL, np.log(np.array(Tstar)-Tc), 1)[0]
print(f"gamma/nu: from chi'_max = {g1:.3f}, from chi(Tc) = {g2:.3f}   (exact 7/4 = 1.75)")
print(f"1/nu: from dU/dT|Tc = {inu1:.3f}, from T*(L)-Tc = {inu2:.3f}   (exact 1)")
nu = 1/inu1; print(f"=> nu = {nu:.3f}, gamma = (gamma/nu)*nu = {g2*nu:.3f} (using chi(Tc)), {g1*nu:.3f} (using chi'_max)")
# pairwise (successive L) estimates show corrections to scaling
for k in range(3):
    print(f"   L={Ls[k]}->{Ls[k+1]}: gamma/nu(chi Tc)={np.log(chiTc[k+1]/chiTc[k])/np.log(2):.3f}, 1/nu(slope)={np.log(slope[k+1]/slope[k])/np.log(2):.3f}")
