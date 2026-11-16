"""Lab 7.3.1 (key 73_attn): a vanilla self-attention model on raw coordinates vs an invariant GNN.
Transformer: token_i = Emb(Z_i) + Linear(x_i - centroid); 2 pre-LN self-attention blocks (d=64, 4 heads);
E = sum_i MLP(token_i); F = -dE/dR by autograd (conservative, translation-invariant, permutation-equivariant,
but with NO rotational symmetry built in).
Part A  raw MD frames (the molecule tumbles freely) vs canonical frames (every frame rotated so that its C,C,O
        best match frame 0, Kabsch) -- how hard is it to learn the orientation from data?
Part B  canonical-frame models, with and without random-rotation augmentation, tested on canonical and on
        randomly rotated test frames; an invariant SchNet trained with the same loop for comparison.
N = 1000 training frames (300 K), 3000 steps, batch 16, Adam + cosine.  Runtime ~12-15 min on one shared
CPU thread (machine-dependent; measured under load from other jobs)."""
import time, json, math
import numpy as np, torch, torch.nn as nn
from importlib.machinery import SourceFileLoader
torch.set_num_threads(1)
U = SourceFileLoader("u72b", "../../ch07/labs/72b_utils.py").load_module()
OUT = "../../ch07/labs"


class Block(nn.Module):
    def __init__(self, d=64, heads=4, ff=128):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.SiLU(), nn.Linear(ff, d))

    def forward(self, x):                                    # x (B, n, d)
        B, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), -1)   # (B,h,n,n)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, n, d))
        return x + self.ff(self.ln2(x))


class XYZTransformer(nn.Module):
    def __init__(self, Zmol, d=64, n_layers=2):
        super().__init__()
        self.register_buffer("Zmol", torch.tensor(Zmol))
        self.emb, self.pos = nn.Embedding(20, d), nn.Linear(3, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.e0 = 0.0

    def forward(self, R):
        x = self.emb(self.Zmol)[None] + self.pos(R - R.mean(1, keepdim=True))
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1).sum(1) + self.e0


def rotate(Q, X):
    return torch.einsum("bxy,bny->bnx", Q, X)




def kabsch(P, Q):
    """Proper rotation M minimising |M(P - cP) - (Q - cQ)| (Kabsch algorithm, det M = +1)."""
    P = P - P.mean(0); Q = Q - Q.mean(0)
    Uu, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ Uu.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ Uu.T


class XYZMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(27, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
        self.e0 = 0.0

    def forward(self, R):
        return self.f((R - R.mean(1, keepdim=True)).reshape(len(R), -1)).squeeze(-1) + self.e0


R, Z, E, F = U.load(300)
tr, va, te = U.split_300K(seed=0)
ref = R[0] - R[0].mean(0)
Rc, Fc = np.empty_like(R), np.empty_like(F)
ang = np.empty(len(R))
for k in range(len(R)):
    M = kabsch(R[k][:3], ref[:3])
    Rc[k] = (R[k] - R[k].mean(0)) @ M.T; Fc[k] = F[k] @ M.T
    ang[k] = np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))
print(f"orientation of raw frames relative to frame 0 (Kabsch on C,C,O): median {np.median(ang):.0f} deg, "
      f"5-95% {np.percentile(ang, 5):.0f}-{np.percentile(ang, 95):.0f} deg")
rng = np.random.default_rng(7)
Qte = torch.as_tensor(U.random_rotations(len(te), rng), dtype=torch.float32)
STEPS, BS, LR, N = 3000, 16, 3e-3, 1000
idx = tr[:N]


def report(m, Rt, Ft):
    Rt, Ft = torch.as_tensor(Rt), torch.as_tensor(Ft)
    eE, eF = U.evaluate(m, Rt.numpy(), E[te], Ft.numpy())
    Rrot = rotate(Qte, Rt)
    eEr, eFr = U.evaluate(m, Rrot.numpy(), E[te], rotate(Qte, Ft).numpy())
    E0, F0 = U.ef(m, Rt); E1, F1 = U.ef(m, Rrot)
    eq = (F1.detach() - rotate(Qte, F0.detach())).abs().mean().item()
    inv = (E1 - E0).detach().abs().mean().item()
    return dict(E=eE * 1e3, F=eF * 1e3, E_rot=eEr * 1e3, F_rot=eFr * 1e3, F_equiv=eq * 1e3, E_inv=inv * 1e3)


runs = [("Transformer, raw frames", XYZTransformer, "raw", False),
        ("MLP(xyz), canonical", XYZMLP, "canon", False),
        ("Transformer, canonical", XYZTransformer, "canon", False),
        ("Transformer, canonical +aug", XYZTransformer, "canon", True),
        ("SchNet, raw frames", None, "raw", False)]
res, hist, models = {}, {}, {}
t_all = time.time()
for name, cls, frame, aug in runs:
    torch.manual_seed(0)
    m = U.SchNetMol(Z, n_feat=32, n_rbf=20, cutoff=5.0, n_inter=3) if cls is None else cls(Z) if cls is XYZTransformer else cls()
    Rs, Fs = (R, F) if frame == "raw" else (Rc, Fc)
    h = []
    t = U.train_ef(m, Rs[idx], E[idx], Fs[idx], Rs[va], E[va], Fs[va], steps=STEPS, bs=BS, lr=LR, augment=aug, history=h)
    res[name] = report(m, Rs[te], Fs[te]); hist[name] = h; models[name] = m
    print(f"{name:28s} params {U.sn.count_params(m):6d}  ({t:.0f} s)")
    if name in ("Transformer, canonical", "SchNet, raw frames"):
        torch.save(m.state_dict(), f"{OUT}/73a_{'transformer_canon' if 'Trans' in name else 'schnet'}_N1000.pt")

print(f"\n{'model':28s} {'E MAE':>7s} {'F MAE':>7s} | {'rotated test: E':>15s} {'F':>7s} | {'|E(Qx)-E(x)|':>12s} {'|F(Qx)-QF(x)|':>13s}")
for k, v in res.items():
    print(f"{k:28s} {v['E']:7.1f} {v['F']:7.1f} | {v['E_rot']:15.1f} {v['F_rot']:7.1f} | {v['E_inv']:12.1f} {v['F_equiv']:13.1f}")
print("units: meV (energy per molecule) and meV/A; test = 800 later frames (in each model's own frame convention);"
      " rotated = one uniform random rotation per test frame; baseline (zero force) F MAE = 712 meV/A")
for k, h in hist.items():
    sel = {s: f for s, e, f in h}
    print(f"val F MAE vs step {k:28s}: " + "  ".join(f"{s}:{sel[s]:.0f}" for s in (100, 500, 1000, 2000, 3000)))

# rotation sweep: rotate all (canonical / raw) test frames by theta about one fixed random axis
axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
thetas = np.linspace(0, 180, 13)
sweep = {}
for k in ["Transformer, canonical", "Transformer, canonical +aug", "SchNet, raw frames"]:
    Rs, Fs = (R, F) if "raw" in k else (Rc, Fc)
    Rt, Ft = torch.as_tensor(Rs[te]), torch.as_tensor(Fs[te])
    out = []
    for th in np.radians(thetas):
        Qb = torch.as_tensor(np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K, dtype=torch.float32).expand(len(te), 3, 3)
        out.append(U.evaluate(models[k], rotate(Qb, Rt).numpy(), E[te], rotate(Qb, Ft).numpy())[1] * 1e3)
    sweep[k] = out
    print(f"sweep {k:28s}: F MAE at 0/15/45/90/180 deg = {out[0]:.0f} / {out[1]:.0f} / {out[3]:.0f} / {out[6]:.0f} / {out[12]:.0f} meV/A")
json.dump({"res": res, "hist": hist, "thetas": thetas.tolist(), "sweep": sweep}, open(f"{OUT}/73a_attn_results.json", "w"), indent=1)
print(f"wall {time.time()-t_all:.0f} s")
