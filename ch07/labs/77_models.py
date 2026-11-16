"""Ch.7.7 case study: the three model families, a common data pipeline and a common trainer.
  DirectMLP      : E = MLP(x - x_centroid) on the 27 raw Cartesian coordinates (translation-invariant only)
  SchNetLike     : invariant continuous-filter GNN on distances (Schutt et al. 2017/2018 style, minimal)
  EquivariantPotential (from 76_e3nn_block.py): NequIP-style, l_max = 1, 2 interaction blocks
All three are conservative: F = -dE/dx by autograd (Ch.1 Sec.1.8.5)."""
import os, math, time, importlib.util
import numpy as np, torch, torch.nn as nn
from e3nn import o3

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
_spec = importlib.util.spec_from_file_location("blk76", os.path.join(HERE, "76_e3nn_block.py"))
blk76 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(blk76)
EquivariantPotential, full_graph = blk76.EquivariantPotential, blk76.full_graph
NAT, SPECIES = 9, {1: 0, 6: 1, 8: 2}


def load(T):
    d = np.load(os.path.join(DATA, f"ethanol_xtb_{T}K.npz"))
    return (torch.as_tensor(d["R"], dtype=torch.float32), torch.as_tensor(d["E"], dtype=torch.float32),
            torch.as_tensor(d["F"], dtype=torch.float32), d["Z"])


class DirectMLP(nn.Module):
    """frame=False: raw centred coordinates (not rotation-invariant).
    frame=True : coordinates expressed in a molecular frame built from C1->C2 and C2->O (Gram-Schmidt),
                 i.e. canonicalisation; E is then exactly rotation-invariant (but not permutation-aware)."""
    def __init__(self, h=128, depth=3, frame=False):
        super().__init__()
        self.frame = frame
        L, d = [], 3 * NAT
        for _ in range(depth): L += [nn.Linear(d, h), nn.SiLU()]; d = h
        self.net = nn.Sequential(*L, nn.Linear(d, 1))

    def forward(self, R):                                   # R: (B, 9, 3)
        x = R - R.mean(1, keepdim=True)                    # removes translation, nothing else
        if self.frame:
            e1 = nn.functional.normalize(R[:, 1] - R[:, 0], dim=-1)
            v = R[:, 2] - R[:, 1]; e2 = nn.functional.normalize(v - (v * e1).sum(-1, keepdim=True) * e1, dim=-1)
            e3 = torch.cross(e1, e2, dim=-1)
            x = torch.einsum("bai,bji->baj", x, torch.stack([e1, e2, e3], 1))   # components along e1, e2, e3
        return self.net(x.reshape(len(R), -1)).squeeze(-1)


def ssp(x): return nn.functional.softplus(x) - math.log(2.0)    # shifted softplus (SchNet)


class SSP(nn.Module):
    def forward(self, x): return ssp(x)


class SchNetLike(nn.Module):
    def __init__(self, F=64, n_int=3, n_rbf=32, rc=5.0, gamma=10.0):
        super().__init__()
        self.emb = nn.Embedding(3, F)
        self.register_buffer("mu", torch.linspace(0, rc, n_rbf)); self.gamma, self.rc = gamma, rc
        self.filt = nn.ModuleList([nn.Sequential(nn.Linear(n_rbf, F), SSP(), nn.Linear(F, F)) for _ in range(n_int)])
        self.in2f = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(n_int)])
        self.f2out = nn.ModuleList([nn.Sequential(nn.Linear(F, F), SSP(), nn.Linear(F, F)) for _ in range(n_int)])
        self.out = nn.Sequential(nn.Linear(F, F // 2), SSP(), nn.Linear(F // 2, 1))

    def forward(self, R, species, src, dst, batch):
        B = len(R); pos = R.reshape(-1, 3)
        r = (pos[dst] - pos[src]).norm(dim=-1)
        rbf = torch.exp(-self.gamma * (r[:, None] - self.mu) ** 2)
        fcut = 0.5 * (torch.cos(math.pi * r / self.rc) + 1) * (r < self.rc)
        h = self.emb(species)
        for filt, a, b in zip(self.filt, self.in2f, self.f2out):
            W = filt(rbf) * fcut[:, None]                   # continuous filter W(r_ij)
            x = a(h)
            m = torch.zeros_like(x).index_add_(0, dst, x[src] * W)    # cfconv: sum_j x_j * W(r_ij)
            h = h + b(m)                                    # residual interaction
        e = self.out(h).squeeze(-1)
        return torch.zeros(B, dtype=e.dtype).index_add_(0, batch, e)


class Wrapped(nn.Module):
    """Uniform interface E(R) with R (B,9,3); handles graphs and the energy shift/scale."""
    def __init__(self, kind, Z, mu=0.0, sigma=1.0):
        super().__init__()
        self.kind = kind
        self.register_buffer("species", torch.as_tensor([SPECIES[z] for z in Z]))
        self.register_buffer("mu", torch.tensor(float(mu))); self.register_buffer("sigma", torch.tensor(float(sigma)))
        if kind.startswith("mlp"): self.core = DirectMLP(frame=(kind == "mlp_frame"))
        elif kind == "schnet": self.core = SchNetLike()
        elif kind == "nequip": self.core = EquivariantPotential(irreps_hidden="16x0e + 8x1o + 8x1e", lmax=1, n_layers=2)
        self._g = {}

    def graph(self, B):
        if B not in self._g: self._g[B] = full_graph(NAT, B)
        return self._g[B]

    def forward(self, R):
        B = len(R)
        if self.kind.startswith("mlp"): e = self.core(R)
        else:
            src, dst, batch = self.graph(B); sp = self.species.repeat(B)
            if self.kind == "schnet": e = self.core(R, sp, src, dst, batch)
            else:
                oh = nn.functional.one_hot(sp, 3).to(R.dtype)
                e = self.core(oh, R.reshape(-1, 3), src, dst, batch, B)
        return self.mu + self.sigma * e


def energy_forces(model, R, create_graph=False):
    R = R.clone().requires_grad_(True)
    E = model(R)
    F = -torch.autograd.grad(E.sum(), R, create_graph=create_graph)[0]
    return E, F


def random_rotations(n, gen):
    q = torch.randn(n, 4, generator=gen); q = q / q.norm(dim=1, keepdim=True)   # uniform unit quaternions
    a, b, c, d = q.T
    return torch.stack([a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c),
                        2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b),
                        2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d], 1).reshape(n, 3, 3)


def train(kind, Rtr, Etr, Ftr, Rva, Eva, Fva, Z, steps, bs=16, lr=3e-3, lam_E=1.0, seed=0, augment=False, log=None):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    sigma = Ftr.pow(2).mean().sqrt().item()                 # force RMS sets the energy scale (NequIP-style)
    model = Wrapped(kind, Z, mu=Etr.mean().item(), sigma=sigma)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    best, best_state, n = np.inf, None, len(Etr)
    for s in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=gen)
        R, E, F = Rtr[idx], Etr[idx], Ftr[idx]
        if augment:                                          # rotate inputs AND force labels (E unchanged)
            Q = random_rotations(len(idx), gen)
            R = torch.einsum("bij,baj->bai", Q, R); F = torch.einsum("bij,baj->bai", Q, F)
        Ep, Fp = energy_forces(model, R, create_graph=True)
        loss = lam_E * (Ep - E).pow(2).mean() + (Fp - F).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (s + 1) % max(steps // 20, 1) == 0 or s == steps - 1:
            v = evaluate(model, Rva, Eva, Fva)
            score = v["F_mae"] + 0.1 * v["E_mae"] / NAT
            if score < best: best, best_state = score, {k: t.clone() for k, t in model.state_dict().items()}
            if log is not None: log.append((s + 1, v["E_mae"], v["F_mae"]))
    model.load_state_dict(best_state)
    return model


def evaluate(model, R, E, F, bs=100):
    Ep, Fp = [], []
    for i in range(0, len(R), bs):
        e, f = energy_forces(model, R[i:i + bs]); Ep.append(e.detach()); Fp.append(f.detach())
    Ep, Fp = torch.cat(Ep), torch.cat(Fp)
    return dict(E_mae=(Ep - E).abs().mean().item(), F_mae=(Fp - F).abs().mean().item(),
                F_max=(Fp - F).abs().max().item())


def rotation_error(model, R, gen):
    """mean |F(QR) - Q F(R)| / mean |F(R)| and mean |E(QR) - E(R)| over frames with random Q."""
    Q = random_rotations(len(R), gen)
    E0, F0 = energy_forces(model, R); RQ = torch.einsum("bij,baj->bai", Q, R)
    E1, F1 = energy_forces(model, RQ)
    F0r = torch.einsum("bij,baj->bai", Q, F0)
    return ((F1 - F0r).abs().mean() / F0.abs().mean()).item(), (E1 - E0).abs().mean().item()
