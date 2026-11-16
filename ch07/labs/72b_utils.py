"""72b_utils.py -- shared data split, ACSF-BPNN baseline and the energy+force training loop
for Labs 7.2.3 / 7.3.x (owner B).  Everything is float32 on CPU with one thread unless noted."""
import time, math
import numpy as np
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA = _os.path.join(_HERE, "..", "data")
sn = SourceFileLoader("sn72b", _os.path.join(_HERE, "72b_schnet.py")).load_module()


def load(T=300):
    d = np.load(f"{DATA}/ethanol_xtb_{T}K.npz")
    return d["R"].astype(np.float32), d["Z"].astype(np.int64), d["E"].astype(np.float64), d["F"].astype(np.float32)


def split_300K(seed=0):
    """Leakage-aware split (cf. Sec. 6.7): frames are 10 fs apart and strongly correlated, so the test
    set is a *later, separate* stretch of the trajectory.  pool = frames 0-1999 (train/val drawn here),
    gap = 2000-2199 (2 ps discarded), test = 2200-2999 (800 frames)."""
    rng = np.random.default_rng(seed)
    pool = rng.permutation(2000)
    val = pool[:100]
    train_pool = pool[100:]
    test = np.arange(2200, 3000)
    return train_pool, val, test


def random_rotations(n, rng):
    """Uniform random proper rotations (QR of Gaussian matrices with sign fix)."""
    A = rng.standard_normal((n, 3, 3))
    Q, Rr = np.linalg.qr(A)
    Q = Q * np.sign(np.einsum("nii->ni", Rr))[:, None, :]
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1
    return Q


# ---------------------------------------------------------------- ACSF + element-wise MLP (BPNN)
class ACSF(nn.Module):
    """Behler G2 (radial) and G4 (angular) functions, Eq. (6.3)-(6.4) of Ch.6, written in torch so that
    forces follow from autograd.  Fixed (non-learned) first layer."""

    def __init__(self, Zmol, rc=5.0, eta2=4.0, rs=np.linspace(0.8, 4.4, 8), eta4=0.005,
                 zetas=(1.0, 4.0), lams=(1.0, -1.0)):
        super().__init__()
        self.rc = rc
        self.Zmol = Zmol
        self.species = sorted(set(Zmol.tolist()))
        self.register_buffer("rs", torch.tensor(rs, dtype=torch.float32))
        self.eta2, self.eta4, self.zetas, self.lams = eta2, eta4, zetas, lams
        n = len(Zmol)
        self.onehot = torch.tensor([[1.0 if Zmol[j] == s else 0.0 for s in self.species] for j in range(n)])
        pairs = [(a, b) for ai, a in enumerate(self.species) for b in self.species[ai:]]
        M = torch.zeros(n, n, len(pairs))
        for j in range(n):
            for k in range(n):
                for p, (a, b) in enumerate(pairs):
                    if {Zmol[j], Zmol[k]} == {a, b} and (Zmol[j], Zmol[k]) in [(a, b), (b, a)]:
                        M[j, k, p] = 1.0
        self.register_buffer("pairmask", M)
        self.n_out = len(self.species) * len(rs) + len(pairs) * len(zetas) * len(lams)

    def fc(self, r):
        return 0.5 * (torch.cos(math.pi * r / self.rc) + 1.0) * (r < self.rc).to(r.dtype)

    def forward(self, R):                                   # R (B,n,3) -> G (B,n,n_out)
        B, n, _ = R.shape
        vec = R[:, None, :, :] - R[:, :, None, :]           # vec[b,i,j] = R_j - R_i
        eye = torch.eye(n, dtype=torch.bool)
        r = torch.sqrt((vec ** 2).sum(-1) + eye.to(R.dtype))   # avoid sqrt(0) on the diagonal
        fc = self.fc(r) * (~eye).to(R.dtype)
        g2 = torch.exp(-self.eta2 * (r[..., None] - self.rs) ** 2) * fc[..., None]   # (B,i,j,nrs)
        G2 = torch.einsum("bijr,js->bisr", g2, self.onehot).reshape(B, n, -1)
        u = vec / r[..., None]
        cos = torch.einsum("bijx,bikx->bijk", u, u)
        rjk = r[:, None, :, :].expand(B, n, n, n)
        rad = torch.exp(-self.eta4 * (r[:, :, :, None] ** 2 + r[:, :, None, :] ** 2 + rjk ** 2))
        cut = fc[:, :, :, None] * fc[:, :, None, :] * fc[:, None, :, :].expand(B, n, n, n)
        offd = (~torch.eye(n, dtype=torch.bool))[None, None]
        base = rad * cut * offd
        feats = []
        for z in self.zetas:
            for lam in self.lams:
                ang = 2.0 ** (1 - z) * torch.clamp(1.0 + lam * cos, min=0.0) ** z * base   # (B,i,j,k)
                feats.append(torch.einsum("bijk,jkp->bip", ang, self.pairmask))
        G4 = torch.stack(feats, -1).reshape(B, n, -1)
        return torch.cat([G2, G4], -1)


class BPNN(nn.Module):
    def __init__(self, Zmol, hidden=64):
        super().__init__()
        self.acsf = ACSF(Zmol)
        self.Zmol = torch.tensor(Zmol)
        self.species = self.acsf.species
        d = self.acsf.n_out
        self.nets = nn.ModuleDict({str(s): nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                                                         nn.Linear(hidden, hidden), nn.SiLU(),
                                                         nn.Linear(hidden, 1)) for s in self.species})
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))
        self.e0 = 0.0

    def fit_scaler(self, R):
        with torch.no_grad():
            G = self.acsf(R).reshape(-1, self.acsf.n_out)
            self.mu.copy_(G.mean(0)); self.sd.copy_(G.std(0) + 1e-6)

    def forward(self, R):                                   # R (B,n,3) -> E (B,)
        G = (self.acsf(R) - self.mu) / self.sd
        eps = torch.zeros(R.shape[0], R.shape[1], dtype=R.dtype)
        for s in self.species:
            m = self.Zmol == s
            eps[:, m] = self.nets[str(s)](G[:, m]).squeeze(-1)
        return eps.sum(1) + self.e0


class SchNetMol(nn.Module):
    """Wrap the edge-list SchNet as R (B,n,3) -> E (B,)."""

    def __init__(self, Zmol, **kw):
        super().__init__()
        self.net = sn.SchNet(**kw)
        self.register_buffer("Zmol", torch.tensor(Zmol))
        self.e0 = 0.0

    def forward(self, R, dense=True):
        if dense:                                       # fast path, identical function
            return sn.dense_forward(self.net, self.Zmol, R) + self.e0
        Z, Rf, ei, batch, nb = sn.make_batch(self.Zmol, R)
        return self.net(Z, Rf, ei, batch, nb) + self.e0


def ef(model, R, create_graph=False):
    R = R.clone().requires_grad_(True)
    E = model(R)
    F = -torch.autograd.grad(E.sum(), R, create_graph=create_graph)[0]
    return E, F


def train_ef(model, R, E, F, Rv, Ev, Fv, steps=3000, bs=32, lr=2e-3, lamE=0.1, seed=0,
             augment=False, log=False, history=None):
    """Adam + cosine decay on  L = lamE*<dE^2> + <|dF|^2>/3 ; keep the weights with the best validation
    score (E_MAE + F_MAE).  augment=True applies a fresh random rotation to every training batch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.e0 = float(E.mean())
    R, E, F = map(torch.as_tensor, (R, E, F))
    E = E.float(); Ev = torch.as_tensor(Ev).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = len(E); best = (1e9, None); t0 = time.time()
    for step in range(steps):
        idx = torch.as_tensor(rng.choice(n, size=min(bs, n), replace=False))
        Rb, Eb, Fb = R[idx], E[idx], F[idx]
        if augment:
            Q = torch.as_tensor(random_rotations(len(idx), rng), dtype=torch.float32)
            Rb = torch.einsum("bxy,bny->bnx", Q, Rb - Rb.mean(1, keepdim=True))
            Fb = torch.einsum("bxy,bny->bnx", Q, Fb)
        Ep, Fp = ef(model, Rb, create_graph=True)
        loss = lamE * ((Ep - Eb) ** 2).mean() + ((Fp - Fb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 100 == 0 or step == steps - 1:
            e_mae, f_mae = evaluate(model, Rv, Ev, Fv)
            score = e_mae + f_mae
            if history is not None:
                history.append((step + 1, e_mae * 1000, f_mae * 1000))
            if score < best[0]:
                best = (score, {k: v.clone() for k, v in model.state_dict().items()})
            if log:
                print(f"  step {step+1:5d} loss {loss.item():.4f} val E {e_mae*1000:.1f} meV F {f_mae*1000:.1f} meV/A")
    model.load_state_dict(best[1])
    return time.time() - t0


def evaluate(model, R, E, F, bs=200):
    eE, eF = [], []
    for s in range(0, len(R), bs):
        Rb = torch.as_tensor(R[s:s + bs])
        Ep, Fp = ef(model, Rb)
        eE.append((Ep.detach().double() - torch.as_tensor(E[s:s + bs]).double()).abs().numpy())
        eF.append((Fp.detach() - torch.as_tensor(F[s:s + bs])).abs().numpy())
    return float(np.concatenate(eE).mean()), float(np.concatenate(eF).mean())
