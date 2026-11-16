"""Lab 7.2.3a (key 72_graph): from structure to graph, permutation equivariance, and the distance-only limit.
Part A  periodic graphs from a pymatgen Structure (Si, cf. Lab 3.1): cutoffs, image edges, multigraph, self-images
Part B  ethanol: bond graph vs cutoff graph; Laplacian spectrum (cf. Ch.1 Sec. 1.5.8)
Part C  permutation equivariance of message passing / invariance of the sum readout (float64)
Part D  expressivity: CH4 tetrahedral vs square-planar -- what a distance-only GNN can and cannot see
Runtime: a few seconds."""
import numpy as np, torch
from collections import Counter
from importlib.machinery import SourceFileLoader
from pymatgen.core import Structure, Lattice
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
sn = SourceFileLoader("sn72b", "../../ch07/labs/72b_schnet.py").load_module()

# ---------------- Part A
print("=== Part A: Si diamond graphs (a = 5.43 A)")
conv = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
prim = conv.get_primitive_structure()
for name, s in [("conventional", conv), ("primitive", prim)]:
    for rc in [2.5, 4.0]:
        ci, ni, img, d = s.get_neighbor_list(rc)
        pairs = Counter(zip(ci.tolist(), ni.tolist()))
        n_self = int((ci == ni).sum())
        print(f"{name:12s} N={len(s)} r_c={rc} A: directed edges {len(ci):4d}, per atom {len(ci)//len(s):2d}, "
              f"distinct (i,j) pairs {len(pairs):3d}, max parallel edges {max(pairs.values())}, "
              f"self-image edges {n_self}, distances {sorted(set(np.round(d, 3).tolist()))}")
ci, ni, img, d = prim.get_neighbor_list(2.5)
print("primitive, r_c=2.5: edges into atom 0 come from atom", sorted(set(ni[ci == 0].tolist())),
      "with image offsets", [tuple(int(v) for v in o) for o in img[ci == 0]])

# ---------------- Part B
print("=== Part B: ethanol graphs (frame 0 of the 300 K xTB trajectory)")
d = np.load("../../ch07/data/ethanol_xtb_300K.npz")
R0, Z = d["R"][0], d["Z"]
D = np.linalg.norm(R0[:, None] - R0[None], axis=-1)
for rc in [1.7, 2.5, 5.0]:
    A = ((D < rc) & (D > 0)).astype(float)
    print(f"r_c={rc} A: directed edges {int(A.sum()):3d} (complete graph = {9*8})")
A = ((D < 1.7) & (D > 0)).astype(float)
L = np.diag(A.sum(1)) - A
ev = np.linalg.eigvalsh(L)
print("bond-graph Laplacian eigenvalues:", np.round(ev, 3), "| # zero modes =", int((ev < 1e-10).sum()))
A16 = ((D < 1.6) & (D > 0)).astype(float); ev16 = np.linalg.eigvalsh(np.diag(A16.sum(1)) - A16)
print(f"with a 1.6 A threshold instead: C-C distance in this frame {D[0,1]:.3f} A -> # zero modes = {int((ev16 < 1e-10).sum())} (graph splits)")

# ---------------- Part C
print("=== Part C: permutation equivariance (float64, untrained SchNet)")
torch.manual_seed(0)
net = sn.SchNet(n_feat=32, n_rbf=20, cutoff=5.0, n_inter=3)
Zt = torch.tensor(Z); Rt = torch.tensor(R0)
ei = sn.full_graph_edges(9, 1)
batch = torch.zeros(9, dtype=torch.long)
h = net.atom_features(Zt, Rt, ei)
E = net(Zt, Rt, ei, batch, 1)
perm = torch.tensor([1, 0, 2, 5, 3, 4, 8, 6, 7])          # swap the carbons, cycle the H's
hp = net.atom_features(Zt[perm], Rt[perm], ei)
Ep = net(Zt[perm], Rt[perm], ei, batch, 1)
print(f"max|h(P x) - P h(x)| = {(hp - h[perm]).abs().max().item():.1e}   |E(Px) - E(x)| = {(Ep - E).abs().item():.1e}")
mean_read = lambda hh: hh.mean(0); max_read = lambda hh: hh.max(0).values
print(f"readouts invariant? sum {torch.allclose(h.sum(0), hp.sum(0))}, mean {torch.allclose(mean_read(h), mean_read(hp))},"
      f" max {torch.allclose(max_read(h), max_read(hp))}; concat (order-dependent) "
      f"{torch.allclose(h.reshape(-1), hp.reshape(-1))}")

# ---------------- Part D
print("=== Part D: CH4, tetrahedral vs square planar, r_CH = 1.09 A")
r = 1.09
tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3) * r
sq = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]) * r
geoms = {"tetrahedral": np.vstack([[0, 0, 0], tet]), "square planar": np.vstack([[0, 0, 0], sq])}
for g, X in geoms.items():
    DD = np.linalg.norm(X[:, None] - X[None], axis=-1)
    print(f"{g:13s}: H-H distances {sorted(set(np.round(DD[1:,1:][np.triu_indices(4,1)],3).tolist()))}")
Zc = torch.tensor([6, 1, 1, 1, 1])
for rc in [1.5, 5.0]:
    out = {}
    for seed in range(3):
        torch.manual_seed(seed)
        m = sn.SchNet(n_feat=32, n_rbf=20, cutoff=rc, n_inter=3)
        for g, X in geoms.items():
            Xt = torch.tensor(X)
            e = sn.full_graph_edges(5, 1, cutoff=rc, R=Xt)
            out.setdefault(seed, {})[g] = (m(Zc, Xt, e, torch.zeros(5, dtype=torch.long), 1).item(), e.shape[1])
    diffs = [abs(out[s]["tetrahedral"][0] - out[s]["square planar"][0]) for s in range(3)]
    print(f"SchNet r_c={rc}: edges tet/sq = {out[0]['tetrahedral'][1]}/{out[0]['square planar'][1]};"
          f" |E_tet - E_sq| over 3 random inits = {', '.join(f'{x:.1e}' for x in diffs)}")
from numpy.polynomial.legendre import legval
print("angular sums at C:  S_l = sum_{j!=k} P_l(cos theta_jk)")
for g, X in geoms.items():
    u = X[1:] / np.linalg.norm(X[1:], axis=1, keepdims=True)
    c = (u @ u.T)[~np.eye(4, dtype=bool)]
    print(f"  {g:13s}: " + "  ".join(f"l={l}: {legval(c, [0]*l+[1]).sum():+.4f}" for l in range(1, 5)))
