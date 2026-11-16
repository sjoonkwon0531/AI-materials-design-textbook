"""Lab 7.3.3 (key 73_forceeq): invariant energy => equivariant (and conservative) autograd forces.
Four force models on ethanol, all in float64:
  (a) SchNet energy (trained in Lab 72_schnet, N=1000) + autograd forces      -> invariant E, equivariant F, conservative
  (b) raw-xyz MLP energy + autograd forces                                    -> conservative, NOT equivariant
  (c) raw-xyz MLP that outputs the 27 force components directly               -> neither
  (d) SchNet features + "direct equivariant" head F_i = sum_j s_ij r_ij/r_ij -> equivariant, NOT conservative
Tests: 200 random O(3) elements (proper and improper) + translations on 20 test frames; antisymmetric part of
the force Jacobian dF/dR (a gradient field has a symmetric Jacobian = -Hessian); work around a closed loop.
Runtime: < 1 min."""
import os, numpy as np, torch, torch.nn as nn
from importlib.machinery import SourceFileLoader
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
U = SourceFileLoader("u72b", "../../ch07/labs/72b_utils.py").load_module()
sn = U.sn
R, Z, E, F = U.load(300)
_, _, te = U.split_300K()
X = torch.tensor(R[te[:20]], dtype=torch.float64)
rng = np.random.default_rng(1)

# (a) trained SchNet, cast to float64
schnet = U.SchNetMol(Z, n_feat=32, n_rbf=20, cutoff=5.0, n_inter=3)
schnet.load_state_dict(torch.load("../../ch07/labs/72b_schnet_ethanol_N1000.pt"))
schnet = schnet.double(); schnet.e0 = float(E[:2000].mean())

torch.manual_seed(0)
class XYZEnergy(nn.Module):                      # (b)
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(27, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(s, R): return s.f((R - R.mean(1, keepdim=True)).reshape(len(R), -1)).squeeze(-1)
class XYZForce(nn.Module):                       # (c)
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(27, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 27))
    def forward(s, R): return s.f((R - R.mean(1, keepdim=True)).reshape(len(R), -1)).reshape(-1, 9, 3)
class DirectEquivariant(nn.Module):              # (d)
    def __init__(s):
        super().__init__(); s.net = sn.SchNet(n_feat=32, n_rbf=20, cutoff=5.0, n_inter=2)
        s.pair = nn.Sequential(nn.Linear(32 + 20, 32), nn.SiLU(), nn.Linear(32, 1)); s.Z = torch.tensor(Z)
    def forward(s, R):
        B = len(R); Zb, Rf, ei, batch, nb = sn.make_batch(s.Z, R)
        h = s.net.atom_features(Zb, Rf, ei); i, j = ei
        vec = Rf[j] - Rf[i]; r = vec.norm(dim=-1)
        sij = s.pair(torch.cat([h[i] + h[j], s.net.rbf(r)], -1)).squeeze(-1)      # symmetric in (i,j), invariant
        Fi = torch.zeros_like(Rf).index_add_(0, i, sij[:, None] * vec / r[:, None])
        return Fi.reshape(B, 9, 3)

models = {"(a) SchNet E -> -dE/dR": ("E", schnet), "(b) xyz-MLP E -> -dE/dR": ("E", XYZEnergy()),
          "(c) xyz-MLP direct F": ("F", XYZForce()), "(d) SchNet direct equiv. F": ("F", DirectEquivariant())}

def forces(kind, m, R):
    if kind == "E":
        Rr = R.clone().requires_grad_(True); Ee = m(Rr)
        return Ee.detach(), -torch.autograd.grad(Ee.sum(), Rr)[0]
    return None, m(R).detach()

Qs = U.random_rotations(200, rng); Qs[100:] *= -1.0          # second half: improper (det = -1)
Qs = torch.tensor(Qs); ts = torch.tensor(rng.normal(0, 5.0, (200, 3)))
summary = {}
print(f"{'model':28s} {'E inv. err':>11s} {'F equiv. err (SO3)':>19s} {'(det -1)':>10s} {'transl.':>9s} {'asym(dF/dR)':>12s} {'loop work':>11s}")
for name, (kind, m) in models.items():
    E0, F0 = forces(kind, m, X)
    scale = F0.abs().mean()
    eE, eF = [], []
    for k in range(200):
        Xg = torch.einsum("xy,bny->bnx", Qs[k], X) + ts[k]
        Eg, Fg = forces(kind, m, Xg)
        eF.append(((Fg - torch.einsum("xy,bny->bnx", Qs[k], F0)).abs().max() / scale).item())
        if Eg is not None: eE.append((Eg - E0).abs().max().item())
    _, Ft = forces(kind, m, X + torch.tensor([3.0, -2.0, 1.0]))
    # Jacobian of the 27 force components w.r.t. the 27 coordinates, frame 0
    x0 = X[:1].clone()
    def fflat(v):
        v = v.reshape(1, 9, 3)
        if kind == "E":
            vv = v.requires_grad_(True) if not v.requires_grad else v
            return -torch.autograd.grad(m(vv).sum(), vv, create_graph=True)[0].reshape(-1)
        return m(v).reshape(-1)
    J = torch.autograd.functional.jacobian(fflat, x0.reshape(-1))
    asym = ((J - J.T).norm() / (J + J.T).norm()).item()
    # work of the force on atom 3 (hydroxyl H) along a closed circle of radius 0.1 A in the xy-plane
    th = torch.linspace(0, 2 * np.pi, 401)
    path = x0.repeat(401, 1, 1); path[:, 3, 0] += 0.1 * torch.cos(th); path[:, 3, 1] += 0.1 * torch.sin(th)
    _, Fp = forces(kind, m, path)
    dr = path[1:, 3] - path[:-1, 3]; Fm = 0.5 * (Fp[1:, 3] + Fp[:-1, 3])
    W = (Fm * dr).sum().item()
    Xc = X - X.mean(1, keepdim=True)
    netF = (F0.sum(1).norm(dim=-1).max() / scale).item()
    torque = (torch.cross(Xc, F0, dim=-1).sum(1).norm(dim=-1).max() / scale).item()
    summary[name] = dict(E_inv=max(eE) if eE else None, F_so3=max(eF[:100]), F_o3m=max(eF[100:]),
                         transl=((Ft - F0).abs().max() / scale).item(), asym=asym, loop_meV=W * 1000,
                         netF=netF, torque=torque)
    print(f"{name:28s} {max(eE) if eE else float('nan'):11.1e} {max(eF[:100]):19.1e} {max(eF[100:]):10.1e}"
          f" {((Ft - F0).abs().max() / scale).item():9.1e} {asym:12.1e} {W*1000:10.2e} meV")
print(f"{'model':28s} {'|sum_i F_i|/<|F|>':>18s} {'|sum_i r_i x F_i|/<|F|> (A)':>28s}")
for name, v in summary.items():
    print(f"{name:28s} {v['netF']:18.1e} {v['torque']:28.1e}")
import json; json.dump(summary, open("../../ch07/labs/73a_equiv_results.json", "w"), indent=1)
print("errors: E in eV (max over 200 group elements x 20 frames); F relative to <|F|>; loop: r = 0.1 A, 400 segments")
