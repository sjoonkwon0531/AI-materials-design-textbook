"""Small from-scratch periodic MLIPs for the Section 10.2 labs (pure PyTorch, one thread).

  InvNet   : SchNet-like invariant message passing. Features are scalars (l = 0) only; edges carry
             distances only.
  EqNet    : PaiNN-like equivariant message passing. Each atom carries scalars (l = 0) and Cartesian
             vectors (l = 1). The update uses |V v_i|^2, which contains sum_jk W(r_ij) W(r_ik) cos(theta_jik),
             so the bond angle enters already in the first layer.
  head='energy' : E = sum_i eps_i + N*E0 ; F = -dE/dR by autograd (conservative).
  head='direct' : same backbone, an energy head for E and a separate equivariant vector head
                  F_i = W v_i that outputs forces directly (not the gradient of anything).

Stress: positions and cell are both multiplied by (I + eps) with a symmetric eps that requires grad;
sigma = (1/V) dE/deps at eps = 0 (ASE sign convention, tensile positive). See Derivation Box in 10.2.1.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from ase.neighborlist import neighbor_list

torch.set_num_threads(1)
DT = torch.float64


def cosine_cutoff(r, rc):
    return 0.5 * (torch.cos(math.pi * r / rc) + 1.0) * (r < rc)


class BesselRBF(nn.Module):
    def __init__(self, n, rc):
        super().__init__()
        self.rc = rc
        self.register_buffer('k', torch.arange(1, n + 1, dtype=DT) * math.pi / rc)

    def forward(self, r):
        r = r.unsqueeze(-1)
        return math.sqrt(2.0 / self.rc) * torch.sin(self.k * r) / r


class Graph:
    """Edges (i, j, integer image shift S) for a batch of periodic cells; built once per structure."""

    def __init__(self, R, cell, rc, skin=0.0):
        from ase import Atoms
        ii, jj, SS, bb = [], [], [], []
        n_at = R.shape[1]
        for b in range(R.shape[0]):
            at = Atoms('Si%d' % n_at, positions=R[b], cell=cell[b], pbc=True)
            i, j, S = neighbor_list('ijS', at, rc + skin)   # skin > 0 only for MD (fixed list)
            ii.append(i + b * n_at); jj.append(j + b * n_at); SS.append(S); bb.append(np.full(len(i), b))
        self.i = torch.as_tensor(np.concatenate(ii)); self.j = torch.as_tensor(np.concatenate(jj))
        self.S = torch.as_tensor(np.concatenate(SS), dtype=DT); self.eb = torch.as_tensor(np.concatenate(bb))
        self.nb, self.n_at = R.shape[0], n_at
        self.batch = torch.arange(self.nb).repeat_interleave(n_at)

    def subset(self, idx):
        """Graph restricted to the cells in idx (re-indexed)."""
        g = Graph.__new__(Graph)
        idx = torch.as_tensor(idx)
        pos = torch.full((self.nb,), -1, dtype=torch.long); pos[idx] = torch.arange(len(idx))
        m = pos[self.eb] >= 0
        nb_new = pos[self.eb[m]]
        g.i = self.i[m] % self.n_at + nb_new * self.n_at
        g.j = self.j[m] % self.n_at + nb_new * self.n_at
        g.S = self.S[m]; g.eb = nb_new; g.nb = len(idx); g.n_at = self.n_at
        g.batch = torch.arange(g.nb).repeat_interleave(self.n_at)
        return g


def edge_vectors(Rflat, cell, g):
    """r_ij = R_j - R_i + S_ij . cell(b)."""
    shift = torch.einsum('ek,ekl->el', g.S, cell[g.eb])
    return Rflat[g.j] - Rflat[g.i] + shift


class InvNet(nn.Module):
    def __init__(self, rc=4.5, F=32, n_rbf=8, T=2):
        super().__init__()
        self.rc = rc
        self.h0 = nn.Parameter(torch.randn(F, dtype=DT) * 0.5)
        self.rbf = BesselRBF(n_rbf, rc)
        self.filt = nn.ModuleList([nn.Sequential(nn.Linear(n_rbf, F), nn.SiLU(), nn.Linear(F, F)) for _ in range(T)])
        self.lin = nn.ModuleList([nn.Linear(F, F) for _ in range(T)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, F)) for _ in range(T)])
        self.out = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 1))
        self.double()

    def features(self, rij, g, n_nodes):
        r = rij.norm(dim=-1)
        w = self.rbf(r) * cosine_cutoff(r, self.rc).unsqueeze(-1)
        h = self.h0.expand(n_nodes, -1)
        for filt, lin, upd in zip(self.filt, self.lin, self.upd):
            m = torch.zeros_like(h).index_add(0, g.i, filt(w) * lin(h)[g.j])
            h = h + upd(m)
        return h, None

    def atom_energy(self, h, v):
        return self.out(h).squeeze(-1)


class EqNet(nn.Module):
    def __init__(self, rc=4.5, F=32, n_rbf=8, T=2):
        super().__init__()
        self.rc, self.F = rc, F
        self.h0 = nn.Parameter(torch.randn(F, dtype=DT) * 0.5)
        self.rbf = BesselRBF(n_rbf, rc)
        self.filt = nn.ModuleList([nn.Linear(n_rbf, 3 * F) for _ in range(T)])
        self.phi = nn.ModuleList([nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 3 * F)) for _ in range(T)])
        self.U = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(T)])
        self.V = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(T)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, 3 * F)) for _ in range(T)])
        self.out = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 1))
        self.fvec = nn.Linear(F, 1, bias=False)          # used only by the direct-force head
        self.double()

    def features(self, rij, g, n_nodes):
        F = self.F
        r = rij.norm(dim=-1); rhat = rij / r.unsqueeze(-1)
        w = self.rbf(r) * cosine_cutoff(r, self.rc).unsqueeze(-1)
        s = self.h0.expand(n_nodes, -1)
        v = torch.zeros(n_nodes, 3, F, dtype=DT)
        for filt, phi, U, V, upd in zip(self.filt, self.phi, self.U, self.V, self.upd):
            x = phi(s)[g.j] * filt(w)                                     # (E, 3F)
            ds, dvv, dvs = x[:, :F], x[:, F:2 * F], x[:, 2 * F:]
            s = s + torch.zeros_like(s).index_add(0, g.i, ds)
            dv = v[g.j] * dvv.unsqueeze(1) + rhat.unsqueeze(-1) * dvs.unsqueeze(1)   # (E, 3, F)
            v = v + torch.zeros_like(v).index_add(0, g.i, dv)
            Uv, Vv = U(v), V(v)
            a = upd(torch.cat([s, (Vv ** 2).sum(1)], -1))
            avv, asv, ass = a[:, :F], a[:, F:2 * F], a[:, 2 * F:]
            v = v + avv.unsqueeze(1) * Uv
            s = s + asv * (Uv * Vv).sum(1) + ass
        return s, v

    def atom_energy(self, s, v):
        return self.out(s).squeeze(-1)


class Potential(nn.Module):
    """Wraps a backbone with an energy head (conservative) or a direct-force head."""

    def __init__(self, backbone, head='energy', e0=0.0, escale=1.0):
        super().__init__()
        self.net, self.head = backbone, head
        self.register_buffer('e0', torch.tensor(float(e0), dtype=DT))       # per-atom reference energy
        self.register_buffer('escale', torch.tensor(float(escale), dtype=DT))

    def forward(self, R, cell, g, stress=False, create_graph=False):
        return predict(self, R, cell, g, stress=stress, create_graph=create_graph)


def predict(model, R, cell, g, stress=False, create_graph=False):
    """Energy, forces (and stress) for a batch. Conservative forces by autograd w.r.t. positions."""
    R = R.clone().requires_grad_(True)
    nb, n_at = g.nb, g.n_at
    eps = None
    Rb, cb = R.reshape(nb, n_at, 3), cell
    if stress:
        eps = torch.zeros(nb, 3, 3, dtype=DT, requires_grad=True)
        sym = 0.5 * (eps + eps.transpose(1, 2))
        Rb = Rb + torch.einsum('bnk,bkl->bnl', Rb, sym)
        cb = cell + torch.einsum('bkm,bml->bkl', cell, sym)
    rij = edge_vectors(Rb.reshape(-1, 3), cb, g)
    s, v = model.net.features(rij, g, nb * n_at)
    eps_i = model.net.atom_energy(s, v) * model.escale + model.e0
    E = torch.zeros(nb, dtype=DT).index_add(0, g.batch, eps_i)
    out = {'E': E}
    if model.head == 'direct':
        out['F'] = model.net.fvec(v).squeeze(-1).reshape(nb, n_at, 3)
        if not create_graph:
            out = {k: x.detach() for k, x in out.items()}
        return out
    wrt = [R] + ([eps] if stress else [])
    grads = torch.autograd.grad(E.sum(), wrt, create_graph=create_graph)
    out['F'] = -grads[0].reshape(nb, n_at, 3)
    if stress:
        vol = torch.det(cell).abs()
        out['S'] = grads[1] / vol[:, None, None]
    if not create_graph:
        out = {k: x.detach() for k, x in out.items()}
    return out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
