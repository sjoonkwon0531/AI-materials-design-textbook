"""Lab 62_desc -- SOAP, MBTR, ACSF (DScribe 2.1) and the sine matrix side by side.

Part A: invariance test (rotation of the periodic cell + translation/wrap + permutation)
Part B: smoothness: descriptor distance vs displacement amplitude, and a path scan
Part C: dimension and cost scaling
Part D: SOAP kernel similarity between local environments (geometry only: one species, NN = 2.55 A)
Part E: radial-only incompleteness: tetrahedral vs square-planar CH4
Runtime: about 1-2 min on 2 cores (n_jobs=1 everywhere).
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ase import Atoms
from ase.build import bulk, fcc111, fcc100
from ase.cluster import Icosahedron
import importlib; importlib.import_module("62_dscribe_compat")   # ase>=3.26 fix, see file
from dscribe.descriptors import SOAP, MBTR, ACSF

np.set_printoptions(precision=3, suppress=True)
RC = 5.0
soap_cu = SOAP(species=["Cu"], r_cut=RC, n_max=8, l_max=6, sigma=0.5, periodic=True)
g2 = [[eta, 0.0] for eta in [0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0, 4.0]]
g4 = [[0.005, z, lam] for z in [1, 2, 4, 16] for lam in [-1, 1]]
acsf_cu = ACSF(species=["Cu"], r_cut=RC, g2_params=g2, g4_params=g4, periodic=True)
mbtr_cu = MBTR(species=["Cu"], geometry={"function": "inverse_distance"},
               grid={"min": 0.1, "max": 0.6, "n": 200, "sigma": 0.01},
               weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
               normalization="l2", periodic=True)
mbtr3_cu = MBTR(species=["Cu"], geometry={"function": "cosine"},
                grid={"min": -1, "max": 1, "n": 200, "sigma": 0.02},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                normalization="l2", periodic=True)


def sine_matrix(Z, R, B):
    Z = np.asarray(Z, float)
    F = (R[:, None, :] - R[None, :, :]) @ np.linalg.inv(B)
    D = np.linalg.norm(np.sin(np.pi * F) ** 2 @ B, axis=-1)
    np.fill_diagonal(D, 1.0)
    M = np.outer(Z, Z) / D
    np.fill_diagonal(M, 0.5 * Z ** 2.4)
    return M


def sine_eig(at):
    return np.sort(np.linalg.eigvalsh(sine_matrix(at.numbers, at.positions, np.array(at.cell))))[::-1]


def sine_sorted(at):
    M = sine_matrix(at.numbers, at.positions, np.array(at.cell))
    p = np.argsort(-np.linalg.norm(M, axis=1), kind="stable")
    return M[np.ix_(p, p)].ravel()


DESC = {
    "SOAP (per atom)": lambda a: soap_cu.create(a),
    "ACSF (per atom)": lambda a: acsf_cu.create(a),
    "MBTR k=2 (global)": lambda a: mbtr_cu.create(a),
    "MBTR k=3 (global)": lambda a: mbtr3_cu.create(a),
    "sine matrix, eigenvalues": sine_eig,
    "sine matrix, sorted rows": sine_sorted,
}

base = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))      # 108 atoms
base.rattle(stdev=0.05, seed=11)
N = len(base)

# ---------- Part A
print(f"=== Part A: rattled Cu fcc 3x3x3 (N={N}); rotate cell+atoms, translate+wrap, permute ===")
rng = np.random.default_rng(5)
perm = rng.permutation(N)
tr = base.copy()
q = Rotation.random(random_state=2)
ax = q.as_rotvec(); ang = np.degrees(np.linalg.norm(ax))
tr.rotate(ang, ax / np.linalg.norm(ax), rotate_cell=True)
tr.positions += np.array([0.77, -1.31, 2.05]); tr.wrap()
tr = tr[perm]
inv = np.argsort(perm)
print(f"  raw Cartesian coordinates (after un-permuting): max|dx| = "
      f"{np.abs(tr.positions[inv] - base.positions).max():.3f} A")
for name, f in DESC.items():
    d0, d1 = f(base), f(tr)
    if "per atom" in name:
        d1 = d1[inv]
    print(f"  {name:26s} dim = {d0.size // (N if 'per atom' in name else 1):6d}   "
          f"max|d| = {np.abs(d1 - d0).max():.1e}   relative = {np.abs(d1 - d0).max() / np.abs(d0).max():.1e}")

# ---------- Part B
print("=== Part B: smoothness. Displace along a fixed random pattern u (RMS per atom = amp) ===")
u = rng.normal(size=(N, 3)); u /= np.sqrt((u ** 2).sum(1).mean())


def moved(amp):
    a = base.copy(); a.positions += amp * u; return a


amps = np.logspace(-4, -0.5, 15)
ref = {k: f(base) for k, f in DESC.items()}
rel = {k: [] for k in DESC}
for amp in amps:
    a = moved(amp)
    for k, f in DESC.items():
        rel[k].append(np.linalg.norm(f(a) - ref[k]) / np.linalg.norm(ref[k]))
for k in DESC:
    sl = np.polyfit(np.log10(amps[:8]), np.log10(rel[k][:8]), 1)[0]
    print(f"  {k:26s} rel. distance at 1e-4 A = {rel[k][0]:.2e}, at 0.316 A = {rel[k][-1]:.2e};"
          f" log-log slope (amp<=0.01 A) = {sl:.2f}")
scan = np.linspace(0, 0.3, 301)
h = scan[1] - scan[0]
keys = ["SOAP (per atom)", "sine matrix, eigenvalues", "sine matrix, sorted rows"]
speed = {k: [] for k in keys}
prev = {k: DESC[k](moved(0.0)) for k in keys}


def roworder(a):
    M = sine_matrix(a.numbers, a.positions, np.array(a.cell))
    return np.argsort(-np.linalg.norm(M, axis=1), kind="stable")


prev_order = roworder(moved(0.0)); nswap = 0
for s in scan[1:]:
    a = moved(s)
    for k in keys:
        cur = DESC[k](a)
        speed[k].append(np.linalg.norm(cur - prev[k]) / h / np.linalg.norm(ref[k]))
        prev[k] = cur
    o = roworder(a); nswap += int(np.any(o != prev_order)); prev_order = o
for k in keys:
    v = np.array(speed[k])
    print(f"  path scan 0-0.3 A, step {h:.3f} A: {k:26s} |dD/ds|/|D| median = {np.median(v):8.3f}/A, max = {v.max():8.3f}/A")
print(f"  sorted-row order changed in {nswap} of {len(scan) - 1} steps")
np.savez("62_desc_smooth.npz", amps=amps, scan=scan[1:],
         **{"rel_" + k.split()[0] + "_" + str(i): np.array(rel[k]) for i, k in enumerate(DESC)},
         **{"speed_" + str(i): np.array(v) for i, v in enumerate(speed.values())})

# ---------- Part C
print("=== Part C: dimension and cost (Cu 3x3x3 rattled, 108 atoms, n_jobs=1, best of 5) ===")


def tbest(f, a, rep=5):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(a); ts.append(time.perf_counter() - t0)
    return min(ts)


rows = []
for nmax in [2, 4, 6, 8, 10, 12]:
    for lmax in [2, 4, 6, 8, 10]:
        s = SOAP(species=["Cu"], r_cut=RC, n_max=nmax, l_max=lmax, sigma=0.5, periodic=True)
        t = tbest(s.create, base) / N * 1e3
        rows.append((nmax, lmax, s.get_number_of_features(), t))
print("  SOAP, 1 species: n_max l_max  dim   ms/atom")
for r in rows:
    if r[1] in (2, 6, 10) or r[0] in (8,):
        print(f"                   {r[0]:5d} {r[1]:5d} {r[2]:5d}  {r[3]:7.3f}")
print("  check dim = n_max(n_max+1)/2 * (l_max+1):",
      all(r[2] == r[0] * (r[0] + 1) // 2 * (r[1] + 1) for r in rows))
spec = ["Cu", "Ni", "Al", "Fe"]
print("  species scaling at n_max=8, l_max=6:  S   SOAP dim   ACSF dim   MBTR k2 dim   MBTR k3 dim")
for S in range(1, 5):
    sp = spec[:S]
    s = SOAP(species=sp, r_cut=RC, n_max=8, l_max=6, sigma=0.5, periodic=True)
    a_ = ACSF(species=sp, r_cut=RC, g2_params=g2, g4_params=g4, periodic=True)
    m2 = MBTR(species=sp, geometry={"function": "inverse_distance"},
              grid={"min": 0.1, "max": 0.6, "n": 200, "sigma": 0.01},
              weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3}, periodic=True)
    m3 = MBTR(species=sp, geometry={"function": "cosine"},
              grid={"min": -1, "max": 1, "n": 200, "sigma": 0.02},
              weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3}, periodic=True)
    print(f"                                       {S}   {s.get_number_of_features():7d}   "
          f"{a_.get_number_of_features():7d}   {m2.get_number_of_features():9d}   {m3.get_number_of_features():9d}")
for name in ["ACSF (per atom)", "MBTR k=2 (global)", "MBTR k=3 (global)", "sine matrix, eigenvalues"]:
    print(f"  time per atom, {name:26s} = {tbest(DESC[name], base) / N * 1e3:.3f} ms")
big = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4)); big.rattle(0.05, seed=3)
for name in ["SOAP (per atom)", "MBTR k=3 (global)", "sine matrix, eigenvalues"]:
    t1, t2 = tbest(DESC[name], base) / N, tbest(DESC[name], big) / len(big)
    print(f"  per-atom cost ratio N=256 vs N=108, {name:26s} = {t2 / t1:.2f}")
np.savez("62_desc_cost.npz", rows=np.array(rows))

# ---------- Part D
print("=== Part D: SOAP kernel k = (p.p'/|p||p'|)^2 between environments (all atoms Cu, NN = 2.55 A) ===")
nn = 2.55
envs = {}
fcc = bulk("Cu", "fcc", a=nn * np.sqrt(2), cubic=True).repeat((3, 3, 3)); envs["fcc"] = (fcc, 0)
hcp = bulk("Cu", "hcp", a=nn, c=nn * np.sqrt(8 / 3)).repeat((4, 4, 3)); envs["hcp"] = (hcp, 0)
bcc = bulk("Cu", "bcc", a=nn * 2 / np.sqrt(3), cubic=True).repeat((4, 4, 4)); envs["bcc"] = (bcc, 0)
sc = bulk("Cu", "sc", a=nn).repeat((4, 4, 4)); envs["sc"] = (sc, 0)
dia = bulk("Cu", "diamond", a=nn * 4 / np.sqrt(3), cubic=True).repeat((2, 2, 2)); envs["diamond"] = (dia, 0)
s111 = fcc111("Cu", size=(4, 4, 4), a=nn * np.sqrt(2), vacuum=8.0, periodic=True)
envs["fcc(111) surf"] = (s111, int(np.argmax(s111.positions[:, 2])))
s100 = fcc100("Cu", size=(4, 4, 4), a=nn * np.sqrt(2), vacuum=8.0, periodic=True)
envs["fcc(100) surf"] = (s100, int(np.argmax(s100.positions[:, 2])))
vac = fcc.copy(); d = vac.get_distances(0, range(len(vac)), mic=True)
nbr = int(np.argsort(d)[1]); del vac[0]; nbr -= 1
envs["fcc vacancy nbr"] = (vac, nbr)
r1 = fcc.copy(); r1.rattle(0.10, seed=4); envs["fcc rattled 0.10"] = (r1, 0)
r3 = fcc.copy(); r3.rattle(0.30, seed=4); envs["fcc rattled 0.30"] = (r3, 0)
ico = Icosahedron("Cu", noshells=3)
cidx = int(np.argmin(np.linalg.norm(ico.positions - ico.positions.mean(0), axis=1)))
dc = np.sort(np.linalg.norm(ico.positions - ico.positions[cidx], axis=1))[1]
ico.positions = ico.positions[cidx] + (ico.positions - ico.positions[cidx]) * nn / dc
ico.center(vacuum=8.0); ico.pbc = True; envs["icosahedron ctr"] = (ico, cidx)
names = list(envs)


def kern(rc, sg, zeta=2):
    sk = SOAP(species=["Cu"], r_cut=rc, n_max=8, l_max=6, sigma=sg, periodic=True)
    P = np.array([sk.create(a, centers=[i])[0] for a, i in envs.values()])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return (P @ P.T) ** zeta


Kwide = kern(5.0, 0.5)
print(f"  r_cut=5.0, sigma=0.5 (Part A-C settings): min K = {Kwide.min():.3f}, fcc-hcp = {Kwide[0, 1]:.4f}, "
      f"fcc-bcc = {Kwide[0, 2]:.4f}, fcc-diamond = {Kwide[0, 4]:.3f}")
K = kern(3.5, 0.3)
print("  r_cut=3.5, sigma=0.3 (sharper, used for the heat map):")
np.savez("62_desc_kernel.npz", K=K, names=np.array(names))
for i, n in enumerate(names):
    print(f"    {n:17s} " + " ".join(f"{K[i, j]:.3f}" for j in range(len(names))))
print(f"  min K = {K.min():.3f}; fcc-hcp = {K[0, 1]:.4f}; fcc-bcc = {K[0, 2]:.4f}; "
      f"fcc-(fcc rattled 0.10) = {K[0, 8]:.4f}")

# ---------- Part E
print("=== Part E: CH4 tetrahedral vs square planar (C-H = 1.09 A), descriptor of the C atom ===")
r = 1.09
tet = r / np.sqrt(3) * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
sq = r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
mols = [Atoms("CH4", positions=np.vstack([[0, 0, 0], x])) for x in (tet, sq)]
a2 = ACSF(species=["C", "H"], r_cut=RC, g2_params=g2)
a24 = ACSF(species=["C", "H"], r_cut=RC, g2_params=g2, g4_params=g4)
sm = SOAP(species=["C", "H"], r_cut=RC, n_max=8, l_max=6, sigma=0.5)
for nm, d in [("ACSF G2 only", a2), ("ACSF G2+G4", a24), ("SOAP", sm)]:
    x, y = d.create(mols[0], centers=[0])[0], d.create(mols[1], centers=[0])[0]
    print(f"  {nm:13s}: ||d_tet - d_sq|| / ||d_tet|| = {np.linalg.norm(x - y) / np.linalg.norm(x):.2e}")
