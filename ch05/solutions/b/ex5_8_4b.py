"""Ex 5.8.4(b) cont.: why is the measured-form-factor spinodal (9.09) below the discrete-Gaussian value (9.94)?
Athermal A5B5 DPD melt (Lab 5.13 setup, 10^3 box, 300 chains): measure intrachain <r_ij^2>(|i-j|) and the block
form factors with good statistics; compare chi_s N from (i) measured form factors, (ii) a Gaussian chain with the
measured <r_ij^2>, (iii) the discrete Gaussian chain with uniform b^2."""
import sys, time, numpy as np
sys.path.insert(0, "../../../ch05/labs")
from dpd_core import seed, forces, run, make_amat
from bcp_common import build_melt, unwrap_chains, intrachain_sq
from scipy.optimize import minimize_scalar
t0 = time.time(); N, NA, L = 10, 5, 10.0
box = np.array([L, L, L]); nch = int(3*L**3/N)
rng = np.random.default_rng(21); seed(21)
x, v, typ, bonds = build_melt(nch, N, 0.5, box, rng)
A = make_amat(25.0, 0.0); f = np.zeros_like(x); pt = np.zeros(3)
forces(x, v, typ, A, 4.5, 3.0, 0.04, box, bonds, 4.0, f, pt)
run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 1000, 10**6, pt)
qf = np.linspace(0.8, 2.4, 161); ff = np.zeros((len(qf), 3)); ffs = []; M = np.zeros(N); M4 = np.zeros(N); ns = 0
for blk in range(100):
    run(x, v, f, typ, A, 4.5, 3.0, 0.04, 0.65, box, bonds, 4.0, 100, 10**6, pt)
    fs = intrachain_sq(x, nch, N, NA, box, qf, nrand=24, rng=rng); ff += fs; ffs.append(fs)
    y = unwrap_chains(x, nch, N, box)
    for k in range(1, N):
        d2 = np.sum((y[:, k:] - y[:, :-k])**2, 2); M[k] += d2.mean(); M4[k] += (d2**2).mean()
    ns += 1
ff /= ns; M /= ns; M4 /= ns
print("<r^2>(k) for |i-j|=k:", " ".join(f"{M[k]:.3f}" for k in range(1, N)))
print("<r^2>(k)/(k <b^2>):    ", " ".join(f"{M[k]/(k*M[1]):.3f}" for k in range(1, N)))
print("non-Gaussianity <r^4>/<r^2>^2 (Gaussian 5/3):", " ".join(f"{M4[k]/M[k]**2:.3f}" for k in range(1, N)))
gam = lambda s: (s[0]+s[1]+2*s[2])/(s[0]*s[1]-s[2]**2)
G = np.array([gam(s) for s in ff]); k = np.argmin(G)
c = np.polyfit(qf[k-6:k+7], G[k-6:k+7], 2); qmin = -c[1]/(2*c[0]); Gmin = np.polyval(c, qmin)
print(f"(i) measured form factors: grid min Gamma at q={qf[k]:.3f} -> chi_s N = {N*G[k]/2:.3f}; parabola fit: q*={qmin:.3f}, chi_s N = {N*Gmin/2:.3f}")
def sgauss(q, R2):
    i = np.arange(N); D = np.abs(i[:, None]-i[None, :]); P = np.exp(-q*q*R2[D]/6)
    a = i < NA; b = ~a
    return P[np.ix_(a, a)].sum()/N, P[np.ix_(b, b)].sum()/N, P[np.ix_(a, b)].sum()/N
R2m = M.copy(); R2m[0] = 0.0
r = minimize_scalar(lambda q: gam(sgauss(q, R2m)), bounds=(0.5, 3), method="bounded", options=dict(xatol=1e-10))
print(f"(ii) Gaussian with measured <r_ij^2>: q*={r.x:.3f}, chi_s N = {N*r.fun/2:.3f}")
R2u = M[1]*np.arange(N)
r = minimize_scalar(lambda q: gam(sgauss(q, R2u)), bounds=(0.5, 3), method="bounded", options=dict(xatol=1e-10))
print(f"(iii) Gaussian with uniform b^2={M[1]:.3f}: q*={r.x:.3f}, chi_s N = {N*r.fun/2:.3f}")
# where do measured and Gaussian(measured r2) form factors differ?
for qq in (1.2, 1.5, 1.8):
    j = np.argmin(abs(qf-qq)); sg = sgauss(qf[j], R2m)
    print(f"  q={qf[j]:.2f}: measured s_AA,s_AB = {ff[j,0]:.4f},{ff[j,2]:.4f}; Gaussian(meas. r2) = {sg[0]:.4f},{sg[2]:.4f}")
# noise bias of a Lab-like estimate: 6 snapshots, minimum over the Lab q grid (step 0.05)
ffs = np.array(ffs); lab = np.isclose((qf*100) % 5, 0) | np.isclose((qf*100) % 5, 5)
est = []
for g0 in range(10):                      # 6 snapshots spaced t=40 apart, as in the Lab
    fm = ffs[g0:g0+60:10].mean(0)[lab]; Gm = np.array([gam(s) for s in fm]); est.append(N*Gm.min()/2)
est = np.array(est)
print(f"Lab-like estimates (6 snapshots, grid step 0.05), 10 groups: mean {est.mean():.3f}, std {est.std(ddof=1):.3f}, "
      f"min {est.min():.3f}, max {est.max():.3f}; all-snapshot grid-min on the same grid {N*np.array([gam(s) for s in ff[lab]]).min()/2:.3f}")
print(f"wall {time.time()-t0:.0f} s")
