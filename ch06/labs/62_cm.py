"""Lab 62_cm -- Coulomb matrix: invariances, permutation strategies, and two failure modes.

Part A: invariance of raw / sorted-row / eigenvalue CM under rotation+translation+permutation (ethanol, ase g2).
Part B: randomized CM (Montavon et al. 2012): spread of the augmented copies.
Part C: discontinuity of the sorted-row CM when two row norms cross (ethanol, O-H stretch).
Part D: non-injectivity of the eigenspectrum: a different 5-atom geometry with the same spectrum.
Part E: periodic crystals: CM of a cell vs sine matrix (Faber et al. 2015) under origin shift + wrap.
CM units: Z in e, distances in Angstrom (M_ij = Z_i Z_j / |R_i - R_j|, M_ii = 0.5 Z_i^2.4).
Runtime: a few seconds.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ase.build import molecule, bulk
from dscribe.descriptors import CoulombMatrix, SineMatrix

rng = np.random.default_rng(7)


def cm(Z, R):
    Z = np.asarray(Z, float)
    D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    np.fill_diagonal(D, 1.0)
    M = np.outer(Z, Z) / D
    np.fill_diagonal(M, 0.5 * Z ** 2.4)
    return M


def sorted_rows(M):
    p = np.argsort(-np.linalg.norm(M, axis=1), kind="stable")  # descending row norm
    return M[np.ix_(p, p)]


def eigs(M):
    return np.sort(np.linalg.eigvalsh(M))[::-1]


# ---------- Part A
mol = molecule("CH3CH2OH")
Z, R = mol.numbers, mol.positions.copy()
N = len(Z)
Q = Rotation.random(random_state=3).as_matrix()
perm = rng.permutation(N)
R2 = (R @ Q.T + np.array([1.7, -3.2, 0.4]))[perm]
Z2 = Z[perm]
M1, M2 = cm(Z, R), cm(Z2, R2)
print(f"=== Part A: ethanol (N={N}), random rotation + translation + permutation ===")
print(f"  raw Cartesian coordinates : max|dx|      = {np.abs(R2 - R).max():.3f} A")
print(f"  raw CM (no sorting)       : max|dM|      = {np.abs(M2 - M1).max():.3f}")
print(f"  sorted-row CM             : max|dM|      = {np.abs(sorted_rows(M2) - sorted_rows(M1)).max():.2e}")
print(f"  eigenvalue spectrum       : max|dlambda| = {np.abs(eigs(M2) - eigs(M1)).max():.2e}")
print("  row norms of M (atoms " + "".join(mol.get_chemical_symbols()) + "): "
      + " ".join(f"{x:.3f}" for x in np.linalg.norm(M1, axis=1)))
cmd = CoulombMatrix(n_atoms_max=N, permutation="sorted_l2")
print(f"  DScribe sorted_l2 vs own  : max diff     = "
      f"{np.abs(cmd.create(mol).reshape(N, N) - sorted_rows(M1)).max():.2e}")

# ---------- Part B
print("=== Part B: randomized CM (sigma = 1), 5 augmented copies of the same ethanol ===")
nrm = np.linalg.norm(M1, axis=1)
base = sorted_rows(M1).ravel()
dists = []
orders = set()
for k in range(5):
    p = np.argsort(-(nrm + rng.normal(0, 1.0, N)), kind="stable")
    orders.add(tuple(p))
    dists.append(np.linalg.norm(M1[np.ix_(p, p)].ravel() - base))
print(f"  distinct row orders = {len(orders)} / 5;  ||M_rand - M_sorted||_F = "
      + ", ".join(f"{x:.1f}" for x in dists) + f"   (||M_sorted||_F = {np.linalg.norm(base):.1f})")

# ---------- Part C
iO = int(np.where(Z == 8)[0][0])
iH = int(np.argmin([np.linalg.norm(R[j] - R[iO]) if (Z[j] == 1) else 9 for j in range(N)]))
u = (R[iH] - R[iO]) / np.linalg.norm(R[iH] - R[iO])
H_idx = [j for j in range(N) if Z[j] == 1 and j != iH]


def geom(s):
    X = R.copy(); X[iH] = R[iH] + s * u; return X


def norms(s):
    return np.linalg.norm(cm(Z, geom(s)), axis=1)


ss = np.linspace(0.0, 1.5, 3001)
gap = np.array([[norms(s)[iH] - norms(s)[j] for j in H_idx] for s in ss])
sign_change = np.where(np.diff(np.sign(gap), axis=0) != 0)
k0, jj = sign_change[0][0], sign_change[1][0]
# refine the crossing by bisection
a, b = ss[k0], ss[k0 + 1]
f = lambda s: norms(s)[iH] - norms(s)[H_idx[jj]]
for _ in range(80):
    c = 0.5 * (a + b)
    a, b = (c, b) if np.sign(f(c)) == np.sign(f(a)) else (a, c)
s_star = 0.5 * (a + b)
print(f"=== Part C: stretch hydroxyl O-H by s (A); its row norm crosses a C-H hydrogen at s* = {s_star:.4f} A ===")
print("     eps      ||sorted(s*+eps)-sorted(s*-eps)||   ||eig(s*+eps)-eig(s*-eps)||")
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    Mp, Mm = cm(Z, geom(s_star + eps)), cm(Z, geom(s_star - eps))
    print(f"  {eps:8.0e}   {np.linalg.norm(sorted_rows(Mp) - sorted_rows(Mm)):12.4f}"
          f"                    {np.linalg.norm(eigs(Mp) - eigs(Mm)):.2e}")
scan = np.linspace(s_star - 0.3, s_star + 0.3, 241)
np.savez("62_cm_scan.npz", s=scan - s_star,
         srt=np.array([sorted_rows(cm(Z, geom(s))).ravel() for s in scan]),
         eig=np.array([eigs(cm(Z, geom(s))) for s in scan]))

# ---------- Part D
print("=== Part D: a 5-atom C1H4 geometry and a different one with the SAME eigenspectrum ===")
Zm = np.array([6, 1, 1, 1, 1])
R0 = molecule("CH4").positions + rng.normal(0, 0.12, (5, 3))       # distorted methane, no symmetry
lam0 = eigs(cm(Zm, R0))


def resid(x):
    return eigs(cm(Zm, x.reshape(5, 3))) - lam0


best = None
for trial in range(20):
    x0 = (molecule("CH4").positions + rng.normal(0, 0.35, (5, 3))).ravel()
    sol = least_squares(resid, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    R1 = sol.x.reshape(5, 3)
    d0 = np.sort(np.linalg.norm(R0[1:] - R0[0], axis=1))
    d1 = np.sort(np.linalg.norm(R1[1:] - R1[0], axis=1))
    hh0 = np.sort([np.linalg.norm(R0[i] - R0[j]) for i in range(1, 5) for j in range(i + 1, 5)])
    hh1 = np.sort([np.linalg.norm(R1[i] - R1[j]) for i in range(1, 5) for j in range(i + 1, 5)])
    diff = max(np.abs(d1 - d0).max(), np.abs(hh1 - hh0).max())
    if np.abs(sol.fun).max() < 1e-10 and (best is None or diff > best[0]):
        best = (diff, R1, d0, d1, hh0, hh1, np.abs(sol.fun).max())
diff, R1, d0, d1, hh0, hh1, res = best
np.savez("62_cm_twins.npz", R0=R0, R1=R1)
print(f"  degrees of freedom 3N-6 = 9 > N = 5 eigenvalues")
print(f"  max|lambda(R1) - lambda(R0)| = {res:.1e}")
print("  sorted C-H distances R0: " + " ".join(f"{x:.3f}" for x in d0))
print("  sorted C-H distances R1: " + " ".join(f"{x:.3f}" for x in d1))
print("  sorted H-H distances R0: " + " ".join(f"{x:.3f}" for x in hh0))
print("  sorted H-H distances R1: " + " ".join(f"{x:.3f}" for x in hh1))
print(f"  largest difference in a sorted distance = {diff:.3f} A  -> different molecules, same descriptor")
print(f"  sorted-row CM distance between them = "
      f"{np.linalg.norm(sorted_rows(cm(Zm, R1)) - sorted_rows(cm(Zm, R0))):.3f}")

# ---------- Part E
def sine_matrix(Z, R, B):
    """Faber et al. (2015): M_ij = Z_i Z_j / |B . sum_k e_k sin^2(pi [B^-1 (R_i-R_j)]_k)|, rows of B = lattice vectors."""
    Z = np.asarray(Z, float)
    F = (R[:, None, :] - R[None, :, :]) @ np.linalg.inv(B)     # fractional separations
    D = np.linalg.norm(np.sin(np.pi * F) ** 2 @ B, axis=-1)
    np.fill_diagonal(D, 1.0)
    M = np.outer(Z, Z) / D
    np.fill_diagonal(M, 0.5 * Z ** 2.4)
    return M


print("=== Part E: rattled NaCl conventional cell (8 atoms, sigma = 0.1 A), shift origin by t and wrap ===")
nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
nacl.rattle(stdev=0.1, seed=1)
Zc, B = nacl.numbers, np.array(nacl.cell)
ref_c, ref_s = eigs(cm(Zc, nacl.positions)), eigs(sine_matrix(Zc, nacl.positions, B))
for t in [np.array([0.3, 0.0, 0.0]), np.array([1.1, 2.0, -0.7]), np.array([2.82, 2.82, 2.82])]:
    sh = nacl.copy(); sh.positions += t; sh.wrap()
    dc = np.abs(eigs(cm(Zc, sh.positions)) - ref_c).max()
    ds = np.abs(eigs(sine_matrix(Zc, sh.positions, B)) - ref_s).max()
    print(f"  t = ({t[0]:5.2f},{t[1]:5.2f},{t[2]:5.2f}) A:  cell-CM eig max|d| = {dc:8.3f}   sine-matrix eig max|d| = {ds:.1e}")
nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)        # perfect cell for the cell-size comparison
ref_s = eigs(sine_matrix(nacl.numbers, nacl.positions, np.array(nacl.cell)))
prim = bulk("NaCl", "rocksalt", a=5.64)
ev_p = eigs(sine_matrix(prim.numbers, prim.positions, np.array(prim.cell)))
print(f"  same crystal, primitive cell (2 atoms): sine eig = {np.round(ev_p, 2)}")
print(f"                conventional cell (8 atoms): sine eig = {np.round(ref_s, 2)}")

# ---------- Part F  (run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 for stable timings)
import time
print("=== Part F: cost of the sine-matrix eigenspectrum vs cell size (Cu fcc supercells, best of 5, 1 BLAS thread) ===")
for rep in [2, 3, 4, 5, 6]:
    at = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((rep, rep, rep)); at.rattle(0.05, seed=1)
    B = np.array(at.cell); ts = []
    for _ in range(5):
        t0 = time.perf_counter(); eigs(sine_matrix(at.numbers, at.positions, B)); ts.append(time.perf_counter() - t0)
    print(f"  N = {len(at):4d}: total {min(ts)*1e3:8.2f} ms, per atom {min(ts)/len(at)*1e3:.4f} ms")
