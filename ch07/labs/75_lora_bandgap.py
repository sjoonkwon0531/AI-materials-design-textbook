"""Lab 75: LoRA from scratch vs full fine-tuning, linear probing and adapters on a band-gap MLP.
Source domain: non-oxide compounds of expt_gap (Magpie 132 features); pretrain an MLP 132-256-256-128-1 (SiLU).
Target domain: pnictogen compounds (formula contains N, P, As or Sb). Fine-tune on N = 50 or 200 of them, test on 568 held out.
Methods (trainable parameters in brackets are printed):
  scratch (new MLP on N oxides only), zero-shot (source model), linear probe (last layer only),
  full fine-tuning, LoRA r = 1, 2, 4, 8 on all four Linear layers (W + (alpha/r) B A, B=0 at init),
  Houlsby-style bottleneck adapters (m = 8) after each hidden layer.
Also: forgetting on the source test set, and the singular-value spectrum of Delta W from full fine-tuning.
Fine-tuning: Adam, up to 400 full-batch steps, lr 1e-3 for every method (no per-method tuning; see text);
20% of the N labelled points are a validation set for early stopping (the starting point counts, so a method can
return the pretrained model unchanged).
5 random target subsets per N (seeds 0-4). Runtime ~1-3 min (shared machine)."""
import time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from pymatgen.core import Composition
torch.set_num_threads(1)
t0 = time.time()
DATA = "../../ch07/data/"

df = pd.read_parquet(DATA + "magpie_expt_gap.parquet")
feat = [c for c in df.columns if c.startswith("MagpieData")]
X = df[feat].values.astype(np.float32); y = df.target.values.astype(np.float32)
is_ox = np.array([bool({str(e) for e in Composition(f).elements} & {"N", "P", "As", "Sb"}) for f in df.formula])   # target domain
rng = np.random.default_rng(0)
src = rng.permutation(np.where(~is_ox)[0]); tgt = rng.permutation(np.where(is_ox)[0])
src_tr, src_te = src[:int(0.8 * len(src))], src[int(0.8 * len(src)):]
tgt_pool, tgt_te = tgt[:400], tgt[400:]
mu, sd = X[src_tr].mean(0), X[src_tr].std(0) + 1e-6          # standardise with SOURCE-train statistics only
Xs = torch.tensor((X - mu) / sd); ys = torch.tensor(y)
print(f"source (no N/P/As/Sb) train {len(src_tr)}, test {len(src_te)} | target (pnictogen) pool {len(tgt_pool)}, test {len(tgt_te)}")
print(f"mean gap: source {y[src].mean():.3f} eV, target {y[tgt].mean():.3f} eV; metals: source {(y[src]==0).mean():.2f}, target {(y[tgt]==0).mean():.2f}; "
      f"target-test MAE of the pool median {np.abs(y[tgt_te]-np.median(y[tgt_pool])).mean():.3f} eV")

def mlp():
    return nn.Sequential(nn.Linear(132, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
def mae(model, idx):
    with torch.no_grad():
        return (model(Xs[idx]).squeeze(-1) - ys[idx]).abs().mean().item()
def n_train(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------- pretraining on the source domain
torch.manual_seed(0)
base = mlp()
opt = torch.optim.AdamW(base.parameters(), lr=2e-3, weight_decay=1e-4)
g = torch.Generator().manual_seed(0)
steps = 3000; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
for s in range(steps):
    b = src_tr[torch.randint(0, len(src_tr), (128,), generator=g).numpy()]
    loss = (base(Xs[b]).squeeze(-1) - ys[b]).abs().mean()      # L1 loss = MAE
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
base_state = {k: v.clone() for k, v in base.state_dict().items()}
print(f"pretrained: params {n_train(base)}, source-test MAE {mae(base, src_te):.3f} eV, "
      f"zero-shot target-test MAE {mae(base, tgt_te):.3f} eV  ({time.time()-t0:.0f} s)")

# ---------------------------------------------------------------- LoRA and adapters, from scratch
class LoRALinear(nn.Module):
    """y = x W^T + b + (alpha/r) x A^T B^T ;  W, b frozen; A ~ N(0, 1/in), B = 0 so Delta W = 0 at start."""
    def __init__(self, lin, r, alpha=None):
        super().__init__()
        self.lin = lin
        for p in self.lin.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, lin.in_features) / lin.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.s = (alpha if alpha is not None else r) / r
        self.enabled = True
    def forward(self, x):
        out = self.lin(x)
        return out + self.s * (x @ self.A.T) @ self.B.T if self.enabled else out
    def delta(self): return self.s * self.B @ self.A
class Adapter(nn.Module):
    """Houlsby et al. 2019 bottleneck: h + W_up f(W_down h), W_up = 0 at init (near-identity)."""
    def __init__(self, d, m):
        super().__init__()
        self.down, self.up = nn.Linear(d, m), nn.Linear(m, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, h): return h + self.up(nn.functional.silu(self.down(h)))

def build(method, seed):
    torch.manual_seed(1000 + seed)
    m = mlp()
    if method == "scratch":
        return m
    m.load_state_dict(base_state)
    if method == "full":
        return m
    for p in m.parameters(): p.requires_grad_(False)
    if method == "zero-shot":
        return m
    if method == "linear probe":
        for p in m[6].parameters(): p.requires_grad_(True)
        return m
    if method.startswith("LoRA"):
        r = int(method.split("=")[1])
        for i in (0, 2, 4, 6):
            m[i] = LoRALinear(m[i], r)
        return m
    if method == "adapter":
        return nn.Sequential(m[0], m[1], Adapter(256, 8), m[2], m[3], Adapter(256, 8), m[4], m[5], Adapter(128, 8), m[6])

def finetune(m, idx, seed, steps=400, lr=1e-3):
    """Adam on L1 loss, full batch; 20% of the N labelled points are held out for early stopping
    (best validation MAE, checked every 10 steps). The test set is never looked at."""
    params = [p for p in m.parameters() if p.requires_grad]
    if not params: return m
    perm = np.random.default_rng(10 + seed).permutation(idx)
    nv = max(len(idx) // 5, 5); va, tr = perm[:nv], perm[nv:]
    opt = torch.optim.Adam(params, lr=lr)
    best = (mae(m, va), {k: v.clone() for k, v in m.state_dict().items()})
    for s in range(steps):
        loss = (m(Xs[tr]).squeeze(-1) - ys[tr]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 10 == 0:
            v = mae(m, va)
            if v < best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()})
    m.load_state_dict(best[1])
    return m

methods = ["scratch", "zero-shot", "linear probe", "full", "LoRA r=1", "LoRA r=2", "LoRA r=4", "LoRA r=8", "adapter"]
res, svals = {}, []
for N in (50, 200):
    print(f"--- N = {N} labelled target compounds (mean +- sd over 5 subsets) ---")
    for meth in methods:
        tm, sm, npar = [], [], None
        for seed in range(5):
            idx = np.random.default_rng(seed).choice(tgt_pool, N, replace=False)
            m = finetune(build(meth, seed), idx, seed)
            npar = n_train(m) if meth != "zero-shot" else 0
            tm.append(mae(m, tgt_te)); sm.append(mae(m, src_te))
            if meth == "full" and N == 200 and seed == 0:
                dW = (m[2].weight - base_state["2.weight"]).detach()
                svals = torch.linalg.svdvals(dW).numpy()
            if meth == "LoRA r=4" and seed == 0 and N == 200:
                for mod in m:
                    if isinstance(mod, LoRALinear): mod.enabled = False
                restored = mae(m, src_te)
        res[f"{N}|{meth}"] = dict(params=npar, tgt=tm, src=sm)
        print(f"  {meth:13s} trainable {npar:7d}  target-test MAE {np.mean(tm):.3f} +- {np.std(tm):.3f}   "
              f"source-test MAE {np.mean(sm):.3f}")
e = svals ** 2 / (svals ** 2).sum()
print(f"Delta W (layer 2, 256x256) from full FT, N=200: top-5 singular values {np.round(svals[:5], 4).tolist()}, "
      f"s1/s10 = {svals[0]/svals[9]:.1f}; energy in top 1/4/8: {e[:1].sum():.2f}/{e[:4].sum():.2f}/{e[:8].sum():.2f}; "
      f"||Delta W||_F/||W||_F = {np.sqrt((svals**2).sum())/base_state['2.weight'].norm().item():.4f}")
print(f"LoRA r=4 (N=200, seed 0) with adapters switched off: source-test MAE {restored:.3f} eV (= pretrained exactly)")
json.dump(res, open("75_lora_results.json", "w")); np.save("75_dW_svals.npy", svals)
print(f"runtime {time.time()-t0:.0f} s")
