"""Lab 74: does pretraining help a small invariant MLIP when expensive labels are scarce?
Target task: GFN2-xTB energies+forces of ethanol at 300 K (fine-tune on N labelled frames, test on 1,000 held-out frames).
Pretraining set: 1,500 geometries from the 600 K and 1000 K trajectories (every 3rd / 2nd frame), used in two ways:
  (S) self-supervised: coordinate denoising (Zaidi et al., ICLR 2023 style) -- no labels at all.
      x~ = x + sigma*eps,  model 'pseudo-force'  -grad E_theta(x~)  is trained to predict  -eps  (points back to x).
  (G) supervised on cheap labels: GFN1-xTB energies+forces on the same 1,500 geometries (a different level of theory).
  (0) scratch: random initialisation.
Model: dense SchNet-like invariant network (adapted from labs/72b_schnet.py, F=32, 2 interactions, 16 RBF, rc=5 A),
forces by autograd. Float32, one thread.  Loss = 0.01 (dE/std E)^2 + (dF/std F)^2 (force-dominated, as is usual for MLIPs).
Pretraining seed 0 (1,500 steps); fine-tuning 800 steps (lr 2e-3, cosine) for every arm, seeds 0,1 for the fine-tune subsets.
All arms get the same 800 fine-tuning steps (pretrained arms have seen 1,500 more steps in total: a known confounder).
Runtime: ~25 min measured under 4-author shared load on 2 cores (~6-8 min on an idle core)."""
import math, time, json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
t_start = time.time()
DATA = "../ch07/data/"

def ssp(x): return nn.functional.softplus(x) - math.log(2.0)
class SSP(nn.Module):
    def forward(self, x): return ssp(x)

class SchNetDense(nn.Module):
    """Invariant: only interatomic distances enter.  E = sum_i eps_i (sum pooling, Lab 6.18)."""
    def __init__(self, Z, F=32, n_rbf=16, rc=5.0, n_inter=2):
        super().__init__()
        self.register_buffer("Z", torch.as_tensor(Z))
        self.rc = rc
        self.emb = nn.Embedding(10, F)
        self.register_buffer("mu", torch.linspace(0, rc, n_rbf))
        self.gamma = 0.5 / (rc / (n_rbf - 1)) ** 2
        self.filt = nn.ModuleList([nn.Sequential(nn.Linear(n_rbf, F), SSP(), nn.Linear(F, F)) for _ in range(n_inter)])
        self.in2f = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(n_inter)])
        self.out = nn.ModuleList([nn.Sequential(nn.Linear(F, F), SSP(), nn.Linear(F, F)) for _ in range(n_inter)])
        self.head = nn.Sequential(nn.Linear(F, F // 2), SSP(), nn.Linear(F // 2, 1))
        nn.init.zeros_(self.head[2].weight); nn.init.zeros_(self.head[2].bias)   # E = shift at init
        self.shift, self.scale = 0.0, 1.0           # E = scale * sum_i eps_i + shift  (set from the training labels)
    def forward(self, R):                            # R (B, n, 3) -> E (B,)
        B, n, _ = R.shape
        vec = R[:, None] - R[:, :, None]
        eye = torch.eye(n, dtype=torch.bool)
        r = torch.sqrt((vec ** 2).sum(-1) + eye.to(R.dtype))
        rbf = torch.exp(-self.gamma * (r[..., None] - self.mu) ** 2)
        fc = 0.5 * (torch.cos(math.pi * r / self.rc) + 1) * (r < self.rc) * (~eye)
        h = self.emb(self.Z)[None].expand(B, n, -1)
        for f, a, o in zip(self.filt, self.in2f, self.out):
            W = f(rbf) * fc[..., None]
            h = h + o((W * a(h)[:, None]).sum(2))
        return self.head(h).squeeze(-1).sum(1) * self.scale / n + self.shift

def energy_forces(model, R, create_graph):
    R = R.clone().requires_grad_(True)
    E = model(R)
    F = -torch.autograd.grad(E.sum(), R, create_graph=create_graph)[0]
    return E, F

def train(model, R, E, F, steps, lr, bs, seed, w_E=0.01, w_F=1.0, denoise_sigma=None):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    n = len(R)
    sE = E.std() if E is not None else None
    sF = F.std() if F is not None else None
    for s in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        if denoise_sigma is not None:                       # self-supervised denoising
            eps = torch.randn(R[idx].shape, generator=g)
            _, Fp = energy_forces(model, R[idx] + denoise_sigma * eps, True)
            loss = ((Fp + eps) ** 2).mean()                 # pseudo-force should point back: -eps
        else:
            Ep, Fp = energy_forces(model, R[idx], True)
            loss = w_E * ((Ep - E[idx]) / sE).pow(2).mean() + w_F * ((Fp - F[idx]) / sF).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

def evaluate(model, R, E, F):
    Es, Fs = [], []
    for k in range(0, len(R), 250):
        e, f = energy_forces(model, R[k:k+250], False)
        Es.append(e.detach()); Fs.append(f.detach())
    Ep, Fp = torch.cat(Es), torch.cat(Fs)
    return (Ep - E).abs().mean().item() * 1e3, (Fp - F).abs().mean().item() * 1e3   # meV, meV/A

def set_scale(model, E, F):
    model.shift = E.mean().item()
    model.scale = 9 * F.std().item()        # crude: per-atom output ~ O(1) in force units

# ------------------------------------------------------------ data
t300 = np.load(DATA + "ethanol_xtb_300K.npz")
Z = t300["Z"]
R300 = torch.tensor(t300["R"], dtype=torch.float32)
E300 = torch.tensor(t300["E"], dtype=torch.float32)
F300 = torch.tensor(t300["F"], dtype=torch.float32)
pool, test = np.arange(0, 2000), np.arange(2000, 3000)      # 300 K: first 2,000 = fine-tune pool, last 1,000 = test
g1 = np.load(DATA + "74_ethanol_gfn1_600_1000K.npz")
Rp = torch.tensor(g1["R"], dtype=torch.float32)
E1 = torch.tensor(g1["E1"], dtype=torch.float32); F1 = torch.tensor(g1["F1"], dtype=torch.float32)
print(f"pretraining geometries: {len(Rp)} (600 K: {(g1['T']==600).sum()}, 1000 K: {(g1['T']==1000).sum()}); "
      f"300 K pool {len(pool)}, test {len(test)}")
print(f"300 K test set: E std {E300[test].std()*1e3:.0f} meV, |F| component MAD from 0 {F300[test].abs().mean()*1e3:.0f} meV/A")

# ------------------------------------------------------------ pretraining (once per seed)
PT_STEPS, FT_STEPS = 1500, 800
pre = {}
for seed in (0,):
    torch.manual_seed(seed)
    m0 = SchNetDense(Z)
    pre[("0", seed)] = {k: v.clone() for k, v in m0.state_dict().items()}
    t0 = time.time()
    torch.manual_seed(seed); mS = SchNetDense(Z); mS.scale = 9.0
    train(mS, Rp, None, None, PT_STEPS, 2e-3, 32, seed, denoise_sigma=0.05)
    pre[("S", seed)] = {k: v.clone() for k, v in mS.state_dict().items()}
    tS = time.time() - t0; t0 = time.time()
    torch.manual_seed(seed); mG = SchNetDense(Z); set_scale(mG, E1, F1)
    train(mG, Rp, E1, F1, PT_STEPS, 2e-3, 32, seed)
    pre[("G", seed)] = {k: v.clone() for k, v in mG.state_dict().items()}
    tG = time.time() - t0
    with torch.no_grad():
        Ez = torch.cat([mG(R300[test][k:k+250]) for k in range(0, 1000, 250)])
    dE = (Ez - Ez.mean()) - (E300[test] - E300[test].mean())     # GFN1 and GFN2 energies differ by a constant
    _, fG = evaluate(mG, R300[test], E300[test], F300[test])
    print(f"seed {seed}: pretrain time S {tS:.0f} s, G {tG:.0f} s;  G zero-shot on GFN2 300 K test: "
          f"E MAE (after removing the mean offset) {dE.abs().mean()*1e3:.0f} meV, F MAE {fG:.0f} meV/A")
torch.save(pre, "74_pretrained_states.pt")

# ------------------------------------------------------------ fine-tuning learning curves
Ns = [50, 200]
res = {}
for N in Ns:
    for init in ("0", "S", "G"):
        for seed in (0, 1):
            rng = np.random.default_rng(100 + seed)
            tr = torch.tensor(rng.choice(pool, N, replace=False))
            m = SchNetDense(Z); m.load_state_dict(pre[(init, 0)])
            m.shift = E300[tr].mean().item()          # re-centre energies on the target labels
            m.scale = {"0": 9 * F300[tr].std().item(), "S": 9.0, "G": 9 * F1.std().item()}[init]
            train(m, R300[tr], E300[tr], F300[tr], FT_STEPS, 2e-3, 32, seed)
            res[(N, init, seed)] = evaluate(m, R300[test], E300[test], F300[test])
        eM = [res[(N, init, s)][0] for s in (0, 1)]; fM = [res[(N, init, s)][1] for s in (0, 1)]
        print(f"N={N:4d} init={init}: E MAE {np.mean(eM):6.1f} ({eM[0]:.1f}, {eM[1]:.1f}) meV   "
              f"F MAE {np.mean(fM):6.1f} ({fM[0]:.1f}, {fM[1]:.1f}) meV/A   [t={time.time()-t_start:.0f}s]")
json.dump({f"{k[0]}_{k[1]}_{k[2]}": v for k, v in res.items()}, open("74_pretrain_results.json", "w"))
print("--- ratio (pretrained / scratch), mean of 2 seeds ---")
for N in Ns:
    s = [np.mean([res[(N, i, k)][j] for k in (0, 1)]) for i in ("0", "S", "G") for j in (0, 1)]
    print(f"N={N:4d}: S/0  E {s[2]/s[0]:.2f}  F {s[3]/s[1]:.2f}   |   G/0  E {s[4]/s[0]:.2f}  F {s[5]/s[1]:.2f}")
print(f"total runtime {time.time()-t_start:.0f} s (shared machine)")
