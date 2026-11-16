"""Lab 73b_bodyorder: what l_max buys (angular resolution) and what it costs.
(A) CH4 tetrahedral vs square planar (the pair of Lab 7.8 Part D, cutoff 1.5 A: only C-H edges, so a distance-only
    GNN cannot tell them apart). One TFN-style layer with spherical harmonics up to l_max gives centre features
    A_l = sum_j R(r_j) Y_l(r_j); their norms |A_l| are invariants (power-spectrum elements, 3-body).
    l_max = 1 is not enough (A_1 = 0 for both: no dipole); l = 2 separates them.
(B) Cost of one TFN/NequIP-style convolution vs l_max (multiplicity 16, 64 random atoms, cutoff 4 A):
    TP paths, weights per edge, forward+backward time (median of 5, one thread, shared machine).
Runtime ~1-2 min."""
import numpy as np, torch, time
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
t0 = time.time()

print("=== Part A: CH4 tetrahedral vs square planar (Lab 7.8 Part D pair), C-H 1.09 A, cutoff 1.5 A ===")
# A distance-only GNN gives identical outputs for the two (owner B's Lab, Part D). Here: one equivariant layer.
# Centre-atom features after one TFN-style layer with scalar inputs: A_l = sum_j R(r_j) Y_l(r_j)  (nu = 1, 2-body);
# the invariant |A_l|^2 is a power-spectrum element (nu = 2, 3-body).
r0 = 1.09
tet = r0 / np.sqrt(3) * torch.tensor([[1., 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
sq = r0 * torch.tensor([[1., 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
for lmax in (1, 2, 3):
    ir_sh = o3.Irreps.spherical_harmonics(lmax)
    tp = o3.FullyConnectedTensorProduct("1x0e", ir_sh, ir_sh, shared_weights=False)   # scalar input x Y -> irreps
    torch.manual_seed(0)
    radial = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, tp.weight_numel))
    inv = {}
    for name, rel in (("tetrahedral", tet), ("square planar", sq)):
        w = radial(soft_one_hot_linspace(rel.norm(dim=1), 0.0, 1.5, 8, basis="smooth_finite", cutoff=True))
        A = tp(torch.ones(4, 1), o3.spherical_harmonics(ir_sh, rel, normalize=True), w).sum(0)   # centre feature
        inv[name] = torch.stack([A[ir_sh.slices()[l]].norm() for l in range(lmax + 1)]).detach()
    print(f"  l_max={lmax}: |A_l| (l=0..{lmax}) tetra = " + " ".join(f"{x:.4f}" for x in inv["tetrahedral"])
          + " | square = " + " ".join(f"{x:.4f}" for x in inv["square planar"])
          + f" | distinguishable: {bool((inv['tetrahedral'] - inv['square planar']).abs().max() > 1e-6)}")

print("=== Part B: cost of one equivariant convolution vs l_max (mult 16, 64 atoms in a 9 A box, rc 4 A) ===")
rng = np.random.default_rng(0)
pos = torch.tensor(rng.uniform(0, 9, (64, 3)))
D = torch.cdist(pos, pos)
src, dst = torch.where((D < 4.0) & (D > 0))
print(f"  edges = {len(src)} (mean {len(src)/64:.1f} neighbours per atom)")
rows = []
for lmax in (0, 1, 2, 3):
    ir_feat = o3.Irreps([(16, (l, (-1) ** l)) for l in range(lmax + 1)])
    ir_sh = o3.Irreps.spherical_harmonics(lmax)
    tp = o3.FullyConnectedTensorProduct(ir_feat, ir_sh, ir_feat, shared_weights=False)
    radial = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.SiLU(), torch.nn.Linear(32, tp.weight_numel))
    torch.manual_seed(0)
    h = ir_feat.randn(64, -1)
    times = []
    for rep in range(5):
        p = pos.clone().requires_grad_(True)
        t1 = time.perf_counter()
        vec = p[src] - p[dst]
        Y = o3.spherical_harmonics(ir_sh, vec, normalize=True)
        w = radial(soft_one_hot_linspace(vec.norm(dim=1), 0.0, 4.0, 8, basis="smooth_finite", cutoff=True))
        out = torch.zeros(64, ir_feat.dim).index_add_(0, dst, tp(h[src], Y, w))
        out.pow(2).sum().backward()
        times.append(time.perf_counter() - t1)
    rows.append((lmax, ir_feat.dim, len(tp.instructions), tp.weight_numel, np.median(times)))
    print(f"  l_max={lmax}: feature dim {ir_feat.dim:4d}, TP paths {len(tp.instructions):3d}, "
          f"weights/edge {tp.weight_numel:6d}, fwd+bwd {1e3*np.median(times):8.1f} ms")
np.save("73b_cost_lmax.npy", np.array(rows))
print(f"  time ratio l_max=3 / l_max=1: {rows[3][4]/rows[1][4]:.1f}x;  l_max=2 / l_max=1: {rows[2][4]/rows[1][4]:.1f}x")
print(f"runtime {time.time()-t0:.1f} s")
