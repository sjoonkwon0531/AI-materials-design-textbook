"""72b_schnet.py -- a minimal SchNet-like invariant GNN in plain PyTorch (Ch.7 Sec. 7.2.3).

Importable module (owner D may copy it):
    from importlib.machinery import SourceFileLoader
    sn = SourceFileLoader("sn", "../ch07/labs/72b_schnet.py").load_module()
    model = sn.SchNet(n_feat=64, n_rbf=32, cutoff=5.0, n_inter=3)
    E, F = sn.energy_forces(model, Z, R, edge_index, batch, n_graphs)

Design (Schuett et al., NeurIPS 2017; J. Chem. Phys. 148, 241722 (2018)):
  * atom embedding h_i^0 = Emb(Z_i)
  * Gaussian radial basis e_k(r) = exp(-gamma (r - mu_k)^2), mu_k on [0, cutoff]
  * continuous-filter convolution  m_i = sum_j  (W1 h_j) * Wfilt(e(r_ij)) * f_c(r_ij)
  * interaction block h_i <- h_i + Dense(ssp(Dense(m_i)))   (residual)
  * atomwise readout eps_i = MLP(h_i),  E = sum_i eps_i   (sum pooling, cf. Lab 6.18)
  * forces F = -dE/dR by autograd (create_graph=True when training on forces)
Only distances enter -> E is invariant to rotations, translations, reflections and
permutations of like atoms by construction.  Edges are an explicit list (i <- j), so
periodic crystals work too if `shifts` (Cartesian image offsets) are passed.
"""
import math
import torch
import torch.nn as nn


def ssp(x):
    """shifted softplus ln(0.5 e^x + 0.5), the SchNet activation (smooth, ssp(0)=0)."""
    return nn.functional.softplus(x) - math.log(2.0)


class SSP(nn.Module):
    def forward(self, x):
        return ssp(x)


def cosine_cutoff(r, rc):
    return 0.5 * (torch.cos(math.pi * r / rc) + 1.0) * (r < rc).to(r.dtype)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=32, cutoff=5.0, gamma=None):
        super().__init__()
        mu = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("mu", mu)
        self.gamma = gamma if gamma is not None else 1.0 / (mu[1] - mu[0]).item() ** 2 / 2.0

    def forward(self, r):
        return torch.exp(-self.gamma * (r[:, None] - self.mu[None, :]) ** 2)


class Interaction(nn.Module):
    def __init__(self, n_feat, n_rbf, cutoff):
        super().__init__()
        self.cutoff = cutoff
        self.filt = nn.Sequential(nn.Linear(n_rbf, n_feat), SSP(), nn.Linear(n_feat, n_feat))
        self.in2f = nn.Linear(n_feat, n_feat, bias=False)
        self.out = nn.Sequential(nn.Linear(n_feat, n_feat), SSP(), nn.Linear(n_feat, n_feat))

    def forward(self, h, rbf, r, edge_index):
        i, j = edge_index                       # message flows j -> i
        W = self.filt(rbf) * cosine_cutoff(r, self.cutoff)[:, None]
        msg = self.in2f(h)[j] * W
        m = torch.zeros_like(h).index_add_(0, i, msg)
        return h + self.out(m)


class SchNet(nn.Module):
    def __init__(self, n_feat=64, n_rbf=32, cutoff=5.0, n_inter=3, max_z=100):
        super().__init__()
        self.cutoff = cutoff
        self.emb = nn.Embedding(max_z, n_feat)
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.blocks = nn.ModuleList([Interaction(n_feat, n_rbf, cutoff) for _ in range(n_inter)])
        self.head = nn.Sequential(nn.Linear(n_feat, n_feat // 2), SSP(), nn.Linear(n_feat // 2, 1))
        self.register_buffer("e_shift", torch.zeros(max_z))   # per-element reference energy
        self.e_scale = 1.0

    def atom_features(self, Z, R, edge_index, shifts=None):
        i, j = edge_index
        vec = R[j] - R[i] if shifts is None else R[j] - R[i] + shifts
        r = torch.sqrt((vec ** 2).sum(-1) + 1e-12)
        rbf = self.rbf(r)
        h = self.emb(Z)
        for blk in self.blocks:
            h = blk(h, rbf, r, edge_index)
        return h

    def forward(self, Z, R, edge_index, batch, n_graphs, shifts=None):
        h = self.atom_features(Z, R, edge_index, shifts)
        eps = self.head(h).squeeze(-1) * self.e_scale + self.e_shift[Z]
        return torch.zeros(n_graphs, dtype=R.dtype, device=R.device).index_add_(0, batch, eps)


def dense_forward(model, Zmol, Rb):
    """Same function as SchNet.forward for B molecules of identical composition Zmol (n,), written with
    dense (B,n,n,.) tensors -- several times faster on CPU for small molecules.  Returns E (B,)."""
    B, n, _ = Rb.shape
    vec = Rb[:, None, :, :] - Rb[:, :, None, :]                 # vec[b,i,j] = R_j - R_i
    eye = torch.eye(n, dtype=torch.bool)
    r = torch.sqrt((vec ** 2).sum(-1) + 1e-12 + eye.to(Rb.dtype))
    rbf = model.rbf(r.reshape(-1)).reshape(B, n, n, -1)
    h = model.emb(Zmol)[None].expand(B, n, -1)
    offd = (~eye).to(Rb.dtype)[None, :, :, None]
    for blk in model.blocks:
        W = blk.filt(rbf) * (cosine_cutoff(r, blk.cutoff)[..., None] * offd)
        m = torch.einsum("bijf,bjf->bif", W, blk.in2f(h))
        h = h + blk.out(m)
    eps = model.head(h).squeeze(-1) * model.e_scale + model.e_shift[Zmol]
    return eps.sum(1)


def energy_forces(model, Z, R, edge_index, batch, n_graphs, shifts=None, create_graph=False):
    """E (n_graphs,) and F = -dE/dR (n_atoms,3) by reverse-mode autodiff (Ch.1 Sec. 1.8)."""
    if not R.requires_grad:
        R = R.clone().requires_grad_(True)
    E = model(Z, R, edge_index, batch, n_graphs, shifts)
    F = -torch.autograd.grad(E.sum(), R, create_graph=create_graph)[0]
    return E, F


# ---------- batching helpers for fixed-size molecules (no torch_geometric needed) ----------
def full_graph_edges(n_atoms, n_mol, cutoff=None, R=None):
    """All ordered pairs i != j inside each molecule (optionally pruned by a cutoff)."""
    ii, jj = torch.meshgrid(torch.arange(n_atoms), torch.arange(n_atoms), indexing="ij")
    keep = ii != jj
    i0, j0 = ii[keep], jj[keep]
    off = (torch.arange(n_mol) * n_atoms)[:, None]
    i = (i0[None, :] + off).reshape(-1)
    j = (j0[None, :] + off).reshape(-1)
    if cutoff is not None and R is not None:
        Rf = R.reshape(-1, 3)
        d = (Rf[j] - Rf[i]).norm(dim=-1)
        m = d < cutoff
        i, j = i[m], j[m]
    return torch.stack([i, j])


def make_batch(Zmol, Rb):
    """Zmol (n_atoms,) long; Rb (B, n_atoms, 3) -> flat tensors for the model."""
    B, n, _ = Rb.shape
    Z = Zmol.repeat(B)
    R = Rb.reshape(B * n, 3)
    batch = torch.arange(B).repeat_interleave(n)
    return Z, R, full_graph_edges(n, B), batch, B


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
