"""Lab 7.6(b): a NequIP-style equivariant interaction block built from e3nn parts, used INSIDE a model.
Block = spherical harmonics of edge directions  x  radial MLP on a Bessel basis (smooth cutoff)
        -> per-edge weighted tensor product (Clebsch-Gordan) -> sum over neighbours
        -> self-interaction (o3.Linear) + residual -> gated nonlinearity (nn.Gate).
The model (EquivariantPotential) stacks such blocks, reads out an invariant energy, and gets forces by autograd.
Importable: the classes are reused by the Ch.7.7 case study (77_*.py).
Run as a script for the equivariance/invariance checks of Lab 7.6(b) (well under 1 min)."""
import math, time
import numpy as np, torch, torch.nn as tnn
from e3nn import o3
from e3nn.nn import Gate, FullyConnectedNet

torch.set_num_threads(1)


class BesselBasis(tnn.Module):
    """e_n(r) = sqrt(2/rc) sin(n pi r / rc) / r times a smooth polynomial envelope that vanishes at rc."""
    def __init__(self, rc=5.0, n=8, p=6):
        super().__init__()
        self.rc, self.p = rc, p
        self.register_buffer("freq", torch.arange(1, n + 1) * math.pi / rc)

    def forward(self, r):
        x = r / self.rc
        p = self.p                                   # polynomial cutoff (Klicpera et al. / NequIP form)
        env = 1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
        env = env * (x < 1)
        return math.sqrt(2 / self.rc) * torch.sin(self.freq * r[:, None]) / r[:, None] * env[:, None]


class InteractionBlock(tnn.Module):
    """One equivariant message-passing layer: h_i <- Gate( Lin(h_i) + Lin( sum_j TP(h_j, Y(r_ij); R(|r_ij|)) ) )."""
    def __init__(self, irreps_in, irreps_hidden, lmax=1, n_basis=8, rc=5.0, avg_neigh=8.0, bug=False):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.bug = bug          # bug=True: feed (1, x/r, y/r, z/r) but LABEL it "4x0e" (a classic mistake)
        self.irreps_sh = o3.Irreps("4x0e") if bug else o3.Irreps.spherical_harmonics(lmax)
        hid = o3.Irreps(irreps_hidden)
        scalars = o3.Irreps([(m, ir) for m, ir in hid if ir.l == 0])
        gated = o3.Irreps([(m, ir) for m, ir in hid if ir.l > 0])
        gates = o3.Irreps(f"{gated.num_irreps}x0e")         # one scalar gate per gated irrep
        self.gate = Gate(scalars, [torch.nn.functional.silu], gates, [torch.sigmoid], gated)
        self.irreps_out = self.gate.irreps_out
        # per-edge tensor product: its weights are produced by the radial MLP (shared_weights=False)
        self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.gate.irreps_in,
                                                 shared_weights=False)
        self.radial = FullyConnectedNet([n_basis, 32, self.tp.weight_numel], torch.nn.functional.silu)
        self.self_int = o3.Linear(self.irreps_in, self.gate.irreps_in)        # self-interaction / residual path
        self.basis = BesselBasis(rc, n_basis)
        self.norm = 1 / math.sqrt(avg_neigh)

    def forward(self, h, edge_src, edge_dst, edge_vec):
        r = edge_vec.norm(dim=-1)
        if self.bug:
            Y = torch.cat([torch.ones_like(r)[:, None], edge_vec / r[:, None]], 1)
        else:
            Y = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization="component")
        w = self.radial(self.basis(r))
        msg = self.tp(h[edge_src], Y, w)
        agg = torch.zeros(h.shape[0], msg.shape[1], dtype=msg.dtype).index_add_(0, edge_dst, msg) * self.norm
        return self.gate(self.self_int(h) + agg)


class EquivariantPotential(tnn.Module):
    """Species embedding (scalars) -> n_layers interaction blocks -> per-atom scalar energy -> sum."""
    def __init__(self, n_species=3, irreps_hidden="16x0e + 8x1o + 8x1e", lmax=1, n_layers=2, rc=5.0, avg_neigh=8.0,
                 bug=False):
        super().__init__()
        n0 = o3.Irreps(irreps_hidden).count("0e")
        self.embed = tnn.Linear(n_species, n0, bias=False)
        blocks, irr = [], o3.Irreps(f"{n0}x0e")
        for _ in range(n_layers):
            b = InteractionBlock(irr, irreps_hidden, lmax=lmax, rc=rc, avg_neigh=avg_neigh, bug=bug)
            blocks.append(b); irr = b.irreps_out
        self.blocks = tnn.ModuleList(blocks)
        self.readout = o3.Linear(irr, "1x0e")                     # only the l=0 part can reach the energy
        self.rc = rc

    def node_features(self, onehot, pos, src, dst):
        h = self.embed(onehot)
        vec = pos[dst] - pos[src]
        feats = []
        for b in self.blocks:
            h = b(h, src, dst, vec); feats.append(h)
        return feats

    def forward(self, onehot, pos, src, dst, batch, n_mol):
        h = self.node_features(onehot, pos, src, dst)[-1]
        e_atom = self.readout(h).squeeze(-1)
        return torch.zeros(n_mol, dtype=e_atom.dtype).index_add_(0, batch, e_atom)


def full_graph(n_atoms, n_mol):
    """All ordered pairs i != j inside each molecule (ethanol: max distance 4.5 A < rc = 5 A)."""
    i, j = np.where(~np.eye(n_atoms, dtype=bool))
    off = np.repeat(np.arange(n_mol) * n_atoms, len(i))
    src = torch.as_tensor(np.tile(i, n_mol) + off); dst = torch.as_tensor(np.tile(j, n_mol) + off)
    batch = torch.as_tensor(np.repeat(np.arange(n_mol), n_atoms))
    return src, dst, batch


if __name__ == "__main__":
    import os, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    t0 = time.time()
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)                    # float64 for the symmetry checks
    d = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "ethanol_xtb_300K.npz"))
    Z = d["Z"]; species = {1: 0, 6: 1, 8: 2}
    onehot1 = torch.nn.functional.one_hot(torch.as_tensor([species[z] for z in Z]), 3).double()
    nm = 4
    pos = torch.as_tensor(d["R"][:nm].reshape(-1, 3)); onehot = onehot1.repeat(nm, 1)
    src, dst, batch = full_graph(9, nm)
    model = EquivariantPotential()
    print("irreps of block outputs:", [str(b.irreps_out) for b in model.blocks])
    blk = model.blocks[1]
    print("block 2: TP paths = %d, TP weights per edge = %d (from radial MLP), block params = %d" % (
        len(blk.tp.instructions), blk.tp.weight_numel, sum(p.numel() for p in blk.parameters())))
    print("model parameters: %d" % sum(p.numel() for p in model.parameters()))

    def energy_forces(p):
        p = p.clone().requires_grad_(True)
        E = model(onehot, p, src, dst, batch, nm)
        F = -torch.autograd.grad(E.sum(), p)[0]
        return E.detach(), F

    R = o3.rand_matrix()                                     # random proper rotation
    E0, F0 = energy_forces(pos)
    E1, F1 = energy_forces(pos @ R.T)
    feats0 = model.node_features(onehot, pos, src, dst)
    feats1 = model.node_features(onehot, pos @ R.T, src, dst)
    D = blk.irreps_out.D_from_matrix(R)
    print("hidden features of block 2: max |h(Rx) - D(R) h(x)| = %.1e  (max |h| = %.2f)" % (
        (feats1[1] - feats0[1] @ D.T).abs().max().item(), feats0[1].abs().max().item()))
    print("energy:  max |E(Rx) - E(x)| = %.1e eV" % (E1 - E0).abs().max())
    print("forces:  max |F(Rx) - R F(x)| = %.1e eV/A" % (F1 - F0 @ R.T).abs().max())
    # inversion: 1o and 1e features transform differently; energy must be invariant (O(3), not only SO(3))
    Pm = -torch.eye(3)
    E2, F2 = energy_forces(pos @ Pm.T)
    feats2 = model.node_features(onehot, pos @ Pm.T, src, dst)
    Dp = blk.irreps_out.D_from_matrix(Pm)
    print("inversion: max |h(-x) - D(P) h(x)| = %.1e, |E(-x)-E(x)| = %.1e, |F(-x)+F(x)| = %.1e" % (
        (feats2[1] - feats0[1] @ Dp.T).abs().max(), (E2 - E0).abs().max(), (F2 + F0).abs().max()))
    t = torch.tensor([1.3, -0.7, 2.1])
    E3, F3 = energy_forces(pos + t)
    print("translation: |dE| = %.1e, |dF| = %.1e" % ((E3 - E0).abs().max(), (F3 - F0).abs().max()))
    perm = torch.as_tensor([0, 1, 2, 3, 4, 5, 7, 6, 8])      # swap two methyl H atoms (6, 7) of molecule 0
    pos_p = pos.clone(); pos_p[:9] = pos[:9][perm]
    E4, F4 = energy_forces(pos_p)
    print("permutation of two H: |dE| = %.1e, |F_perm - P F| = %.1e" % (
        (E4 - E0).abs().max(), (F4[:9] - F0[:9][perm]).abs().max()))
    # what goes wrong if the radial cutoff is not smooth: compare energy just inside/outside rc for a dimer
    print("Bessel*envelope (n=1) at r = 4.5, 4.9, 4.99, 5.0 A:", ["%.1e" % v for v in BesselBasis()(torch.tensor([4.5, 4.9, 4.99, 5.0]))[:, 0].tolist()])
    # the classic bug: Cartesian direction components declared as scalars -> e3nn cannot protect you
    torch.manual_seed(0); bad = EquivariantPotential(bug=True)
    pr = pos.clone().requires_grad_(True); Eb0 = bad(onehot, pr, src, dst, batch, nm)
    Fb0 = -torch.autograd.grad(Eb0.sum(), pr)[0]
    pr2 = (pos @ R.T).requires_grad_(True); Eb1 = bad(onehot, pr2, src, dst, batch, nm)
    Fb1 = -torch.autograd.grad(Eb1.sum(), pr2)[0]
    print("bug (xyz labelled 4x0e): |E(Rx)-E(x)| = %.1e eV (spread of E over 4 frames %.1e), "
          "|F(Rx)-R F(x)| / |F| = %.2f" % ((Eb1 - Eb0).abs().max().item(), (Eb0.max() - Eb0.min()).item(),
                                           ((Fb1 - Fb0 @ R.T).norm() / Fb0.norm()).item()))
    # float32 check
    torch.set_default_dtype(torch.float32)
    model32 = EquivariantPotential(); model32.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    p32 = pos.float().requires_grad_(True); oh32 = onehot.float()
    Ea = model32(oh32, p32, src, dst, batch, nm); Fa = -torch.autograd.grad(Ea.sum(), p32)[0]
    R32 = R.float(); p32b = (pos.float() @ R32.T).requires_grad_(True)
    Eb = model32(oh32, p32b, src, dst, batch, nm); Fb = -torch.autograd.grad(Eb.sum(), p32b)[0]
    print("float32: |E(Rx)-E(x)| = %.1e, |F(Rx)-R F(x)| = %.1e" % ((Eb - Ea).abs().max(), (Fb - Fa @ R32.T).abs().max()))
    print("wall %.1f s" % (time.time() - t0))
