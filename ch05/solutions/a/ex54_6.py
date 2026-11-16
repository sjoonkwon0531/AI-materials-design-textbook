"""Ex 5.4.6: multiscale coarse-graining (force matching) for the 2D LJ dimer fluid of Lab 5.6.
500 fine-MD snapshots (seed 1, same settings, every 40 steps after 15000), bead force = sum of atomic forces
(intramolecular bond forces cancel). F(r) = sum_k c_k hat_k(r), knots 0.8..3.5 step 0.05, linear least squares."""
import time
import numpy as np
from numba import njit
from ibi_chain import get_chain, ibi, NMOL, T, DT, L

KN = np.round(np.arange(0.8, 3.5 + 1e-9, 0.05), 10); K = KN.size; H = 0.05

@njit(cache=True)
def fine_snapshots(N, L, nequil, nframes, every, dt, gamma, seed, T):
    np.random.seed(seed)
    n = 2 * N; side = int(np.ceil(np.sqrt(N))); a = L / side
    x = np.zeros((n, 2)); m = 0
    for ix in range(side):
        for iy in range(side):
            if m < N:
                th = 2 * np.pi * np.random.random(); cx = (ix + 0.5) * a; cy = (iy + 0.5) * a
                x[2*m, 0] = cx + 0.4 * np.cos(th); x[2*m, 1] = cy + 0.4 * np.sin(th)
                x[2*m+1, 0] = cx - 0.4 * np.cos(th); x[2*m+1, 1] = cy - 0.4 * np.sin(th); m += 1
    v = np.random.normal(0.0, np.sqrt(T), (n, 2))
    F, W = ibi.fine_forces(x, L, N)
    c1 = np.exp(-gamma * dt); c2 = np.sqrt(1 - c1 * c1) * np.sqrt(T)
    R = np.zeros((nframes, N, 2)); f = np.zeros((nframes, N, 2)); k = 0
    for step in range(nequil + nframes * every):
        v += 0.5 * dt * F; x += 0.5 * dt * v
        v = c1 * v + c2 * np.random.normal(0.0, 1.0, (n, 2))
        x += 0.5 * dt * v; x %= L
        F, W = ibi.fine_forces(x, L, N)
        v += 0.5 * dt * F
        if step >= nequil and (step - nequil) % every == 0 and k < nframes:
            R[k] = ibi.com(x, L, N)
            for mm in range(N):
                f[k, mm, 0] = F[2*mm, 0] + F[2*mm+1, 0]; f[k, mm, 1] = F[2*mm, 1] + F[2*mm+1, 1]
            k += 1
    return R, f

@njit(cache=True)
def normal_eq(R, f, L, k0, h, K):
    G = np.zeros((K, K)); b = np.zeros(K); ff = 0.0; rmin = 1e9
    nfr, N = R.shape[0], R.shape[1]
    for t in range(nfr):
        A = np.zeros((N, 2, K))
        for i in range(N):
            for j in range(i + 1, N):
                dx = R[t, i, 0] - R[t, j, 0]; dy = R[t, i, 1] - R[t, j, 1]
                dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
                r = np.sqrt(dx * dx + dy * dy)
                if r < rmin: rmin = r
                if r >= k0 + h * (K - 1):
                    continue
                s = (r - k0) / h
                if s < 0: s = 0.0                      # never happens here (rmin > 0.9)
                q = int(s); w = s - q
                ex = dx / r; ey = dy / r
                for (kk, ww) in ((q, 1 - w), (q + 1, w)):
                    if kk < K:
                        A[i, 0, kk] += ww * ex; A[i, 1, kk] += ww * ey
                        A[j, 0, kk] -= ww * ex; A[j, 1, kk] -= ww * ey
        for i in range(N):
            for c in range(2):
                for a in range(K):
                    if A[i, c, a] != 0.0:
                        b[a] += A[i, c, a] * f[t, i, c]
                        for bb in range(K):
                            G[a, bb] += A[i, c, a] * A[i, c, bb]
                ff += f[t, i, c] ** 2
    return G, b, ff, rmin

t0 = time.time()
ch = get_chain()
g_t, P_fine = ch["g_t"], float(ch["P_fine"])
R, f = fine_snapshots(NMOL, L, 15000, 500, 40, DT, 1.0, 1, T)
print(f"snapshots: {R.shape[0]} frames x {R.shape[1]} beads ({time.time()-t0:.0f} s)")
G, b, ff, rmin = normal_eq(R, f, L, 0.8, H, K)
nrow = R.shape[0] * R.shape[1] * 2
act = np.diag(G) > 0                      # knots below the closest observed distance carry no data
coef = np.zeros(K)
coef[act] = np.linalg.solve(G[np.ix_(act, act)], b[act])
chi2 = (ff - 2 * coef @ b + coef @ G @ coef) / nrow
print(f"min bead distance in data = {rmin:.3f};  rows = {nrow}, unknowns = {K} ({act.sum()} with data, first active knot {KN[act][0]:.2f}), "
      f"cond(G_active) = {np.linalg.cond(G[np.ix_(act, act)]):.2e}")
print(f"<|f|^2>/component = {ff/nrow:.4f};  residual chi2 = {chi2:.4f};  explained fraction = {1-chi2/(ff/nrow):.4f}")
# potential on the Lab grid
r = ibi.RGRID
k0 = KN[act][0]
Fr = np.interp(r, KN, coef, right=0.0); Fr[r < k0] = np.nan
ok = ~np.isnan(Fr)
V = np.zeros_like(r)
Vk = np.concatenate([np.cumsum((0.5 * (coef[1:] + coef[:-1]) * H)[::-1])[::-1], [0.0]])   # V(knot_k) = int_k^3.5 F
V[ok] = np.interp(r[ok], KN, Vk)
print('  F at first knots:', np.round(coef[act][:4], 3))
# short-range extrapolation below the data (r < 0.8), as in the Lab's finalize(): steep linear wall
i0 = np.argmax(ok); slope = min((V[i0+5] - V[i0]) / (5 * ibi.DR), -20.0)
V[:i0] = V[i0] + slope * (r[:i0] - r[i0])
Ftab = -np.gradient(V, ibi.DR)
V_ibi = ch["V"]
for x in (1.0, 1.2, 1.5, 2.0, 2.3, 3.0):
    k = np.argmin(np.abs(r - x))
    print(f"  r={x:.1f}: F_MSCG={Fr[k]:+.4f}  V_MSCG={V[k]:+.4f}  V_IBI={V_ibi[k]:+.4f}")
print(f"  well: MS-CG min {V[r>0.9].min():.4f} at {r[r>0.9][np.argmin(V[r>0.9])]:.2f}; IBI min {V_ibi.min():.4f} at {r[np.argmin(V_ibi)]:.2f}")
# CG MD with the MS-CG potential (Lab settings), two seeds
for sd in (100, 101):
    hc, nf, P, _ = ibi.run_cg(NMOL, L, Ftab, 5000, 30000, DT, 1.0, sd, 15, 2.0)
    g = ibi.gofr(hc, nf, NMOL, L)
    print(f"  CG with MS-CG potential, seed {sd}: g RMSE = {ibi.gerr(g, g_t):.4f}, P_CG = {P:.4f}   (P_fine = {P_fine:.4f};"
          f" IBI: RMSE 0.0180, P 0.4434; IBI+P: 0.0156, 0.3223); first peak g = {g.max():.3f} (target {g_t.max():.3f})")
# force-matching residual of the IBI potential (how well IBI matches mean forces)
FI = np.interp(KN, r, -np.gradient(V_ibi, ibi.DR))
chi2_ibi = (ff - 2 * FI @ b + FI @ G @ FI) / nrow
print(f"  force-matching residual with the IBI force: chi2 = {chi2_ibi:.4f}  (MS-CG optimum {chi2:.4f})")
print(f"total {time.time()-t0:.0f} s")
