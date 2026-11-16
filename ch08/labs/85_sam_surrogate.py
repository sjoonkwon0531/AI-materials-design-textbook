"""85_sam_surrogate.py -- Ch.8 Sec.8.5 Lab (b): does a SAM-trained surrogate resist OOD exploitation?

Model-based optimisation loop:
  historical runs (open-loop 12-step schedules near 650-720 K, one noisy episode each)
  -> surrogate J_hat(schedule) (MLP)  -> optimiser maximises J_hat over the whole box [600, 800]^12
  -> evaluate the proposed schedule on the TRUE simulator (2,000 episodes).
Compared surrogate trainers: AdamW, SAM (rho = 0.05, 0.2), ASAM (rho = 0.5), 5-member AdamW ensemble used with a
lower confidence bound (mu - 2 sigma), and AdamW + a trust region (optimiser confined to the data's per-step range).
Run: python 85_sam_surrogate.py   (~4-6 min single thread)
"""
import importlib, importlib.util, copy, sys, os, time
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
E = importlib.import_module("85_film_env")
P = E.FilmParams(); t_start = time.time()


class SAM(torch.optim.Optimizer):
    """Minimal SAM / ASAM wrapper (same algorithm as Sec.8.3, Derivation Box on SAM)."""
    def __init__(self, params, base, rho=0.05, adaptive=False, eta=0.01, **kw):
        super().__init__(params, dict(rho=rho, **kw))
        self.base = base(self.param_groups, **kw); self.param_groups = self.base.param_groups
        self.adaptive, self.eta = adaptive, eta

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad(): loss = closure()                    # gradient at w
        ps = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        T = {p: (p.abs() + self.eta) if self.adaptive else torch.ones_like(p) for p in ps}
        norm = torch.sqrt(sum(((T[p] * p.grad) ** 2).sum() for p in ps)) + 1e-12
        eps = {p: self.param_groups[0]["rho"] * T[p] ** 2 * p.grad / norm for p in ps}
        for p in ps: p.add_(eps[p])                                   # climb to w + eps
        with torch.enable_grad(): closure()                           # gradient at w + eps
        for p in ps: p.sub_(eps[p])                                   # return to w
        self.base.step(); return loss


def mlp():
    return nn.Sequential(nn.Linear(P.N, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))


def train(X, y, seed, mode="adamw", rho=0.05, epochs=400, lr=3e-3, wd=1e-4):
    torch.manual_seed(seed); net = mlp()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    if mode == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = SAM(net.parameters(), torch.optim.AdamW, rho=rho, adaptive=(mode == "asam"), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        for s in range(0, len(yt), 64):
            i = perm[s:s + 64]
            def closure():
                opt.zero_grad(); l = ((net(Xt[i]).squeeze(-1) - yt[i]) ** 2).mean(); l.backward(); return l
            if mode == "adamw": closure(); opt.step()
            else: opt.step(closure)
    return net.eval()


_s83 = importlib.util.spec_from_file_location("sam83", os.path.join(HERE, "83_sam.py"))
S83 = importlib.util.module_from_spec(_s83); _s83.loader.exec_module(S83)


def lambda_max(net, X, y):
    """Largest Hessian eigenvalue of the training MSE w.r.t. weights: Lanczos (scipy eigsh, which='LA') on
    Hessian-vector products of a float64 copy, i.e. 83_sam.lambda_max (Sec.8.3.2, Derivation Box 8.9)."""
    net = copy.deepcopy(net).double(); Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    return S83.lambda_max(lambda: ((net(Xt).squeeze(-1) - yt) ** 2).mean(), list(net.parameters()))[0]


def maximise(score_fn, lo, hi, seed, n_start=64, steps=300, lr=0.03):
    """Projected Adam ascent on the surrogate from n_start random points inside [lo, hi]; best predicted wins."""
    g = torch.Generator().manual_seed(seed)
    x = (lo + (hi - lo) * torch.rand(n_start, P.N, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); (-score_fn(x).sum()).backward(); opt.step()
        with torch.no_grad(): x.copy_(torch.max(torch.min(x, hi), lo))
    with torch.no_grad(): s = score_fn(x)
    return x[int(torch.argmax(s))].detach().numpy(), float(s.max())


def true_J(sched, n=2000, seed=777):
    return E.rollout_batch(lambda t, o: np.full(len(o), sched[t]), n, seed, P)[0].mean()


# ---------------------------------------------------------------- data: "historical runs"
rng = np.random.default_rng(0)
n_tr = 400
base = rng.uniform(650, 720, n_tr)[:, None] + rng.normal(0, 10, (n_tr, P.N))   # K
X = (base - 700) / 100                                                           # action units
y = np.array([E.rollout_batch(lambda t, o, s=s: np.full(1, s[t]), 1, 10_000 + i, P)[0][0] for i, s in enumerate(X)])
ymu, ysd = y.mean(), y.std(); yn = (y - ymu) / ysd
Xid = (rng.uniform(650, 720, 300)[:, None] + rng.normal(0, 10, (300, P.N)) - 700) / 100        # in-distribution
Xood = rng.uniform(-1, 1, (300, P.N))                                                          # whole box
Jid = np.array([true_J(s, 200, 5) for s in Xid]); Jood = np.array([true_J(s, 200, 5) for s in Xood])
print(f"training runs: {n_tr}; label J range [{y.min():.2f}, {y.max():.2f}]; data T range "
      f"[{base.min():.0f}, {base.max():.0f}] K; best training label {y.max():.3f}")
J_twolevel = E.rollout_batch(lambda t, o: np.full(len(o), ((670 if t < 6 else 690) - 700) / 100), 2000, 777, P)[0].mean()
print(f"reference: tuned two-level schedule (Lab 8.14) true J = {J_twolevel:.3f}")
J_hist = true_J(X[int(np.argmax(y))])
print(f"reference: best historical run (label {y.max():.3f}) re-run 2,000 times: true J = {J_hist:.3f}")

lo_box, hi_box = -torch.ones(P.N), torch.ones(P.N)
lo_tr, hi_tr = torch.tensor(X.min(0), dtype=torch.float32), torch.tensor(X.max(0), dtype=torch.float32)

def evaluate(name, nets, score, lo, hi, seed):
    f = lambda Z: np.mean([n(torch.tensor(Z, dtype=torch.float32)).squeeze(-1).detach().numpy() for n in nets], 0) * ysd + ymu
    mae_id, mae_ood = np.abs(f(Xid) - Jid).mean(), np.abs(f(Xood) - Jood).mean()
    x_opt, s_opt = maximise(score, lo, hi, seed)
    pred = float(f(x_opt[None])[0]); tj = true_J(x_opt)
    dmin = np.sqrt(((X - x_opt) ** 2).sum(1)).min() * 100                          # K, Euclidean over 12 steps
    return dict(mae_id=mae_id, mae_ood=mae_ood, pred=pred, true=tj, gap=pred - tj, dist=dmin,
                Tmean=float(E.to_T(x_opt).mean()), Tmin=float(E.to_T(x_opt).min()), Tmax=float(E.to_T(x_opt).max()))

methods = ["AdamW", "SAM rho=0.05", "SAM rho=0.2", "ASAM rho=0.5", "Ensemble-5 LCB", "AdamW + trust region",
           "SAM rho=0.2 + trust region", "Ensemble LCB + trust region"]
R = {m: [] for m in methods}; LAM = {m: [] for m in methods[:4]}
for seed in range(5):
    nets = {}
    nets["AdamW"] = train(X, yn, seed)
    nets["SAM rho=0.05"] = train(X, yn, seed, "sam", 0.05)
    nets["SAM rho=0.2"] = train(X, yn, seed, "sam", 0.2)
    nets["ASAM rho=0.5"] = train(X, yn, seed, "asam", 0.5)
    for m in methods[:4]:
        LAM[m].append(lambda_max(nets[m], X, yn))
        R[m].append(evaluate(m, [nets[m]], lambda Z, n=nets[m]: n(Z).squeeze(-1), lo_box, hi_box, seed))
    ens = [nets["AdamW"]] + [train(X, yn, 100 * seed + k) for k in range(1, 5)]
    def lcb(Z, ens=ens):
        Pm = torch.stack([n(Z).squeeze(-1) for n in ens]); return Pm.mean(0) - 2.0 * Pm.std(0)
    R["Ensemble-5 LCB"].append(evaluate("ens", ens, lcb, lo_box, hi_box, seed))
    R["AdamW + trust region"].append(evaluate("tr", [nets["AdamW"]], lambda Z, n=nets["AdamW"]: n(Z).squeeze(-1),
                                              lo_tr, hi_tr, seed))
    R["SAM rho=0.2 + trust region"].append(evaluate("sam-tr", [nets["SAM rho=0.2"]],
                                       lambda Z, n=nets["SAM rho=0.2"]: n(Z).squeeze(-1), lo_tr, hi_tr, seed))
    R["Ensemble LCB + trust region"].append(evaluate("ens-tr", ens, lcb, lo_tr, hi_tr, seed))
    print(f"  seed {seed} done ({time.time() - t_start:.0f} s)")

print("\nmean +/- sd over 5 seeds; J in units of the reward (higher is better); dist = nearest training run [K]")
print(f"{'surrogate / optimiser':<28s} {'lam_max':>8s} {'MAE id':>7s} {'MAE ood':>8s} {'pred J':>7s} {'true J':>14s} "
      f"{'gap':>12s} {'dist':>6s} {'T range of optimum':>18s}")
for m in methods:
    r = R[m]; a = lambda k: np.array([q[k] for q in r])
    lam = f"{np.mean(LAM[m]):8.2f}" if m in LAM else f"{'-':>8s}"
    print(f"{m:<28s} {lam} {a('mae_id').mean():7.3f} {a('mae_ood').mean():8.2f} {a('pred').mean():7.2f} "
          f"{a('true').mean():7.2f}+/-{a('true').std(ddof=1):4.2f} {a('gap').mean():6.2f}+/-{a('gap').std(ddof=1):4.2f} "
          f"{a('dist').mean():6.0f}   {a('Tmin').mean():4.0f}-{a('Tmax').mean():4.0f} K")
np.savez(os.path.join(HERE, "85_sam_surrogate_results.npz"),
         **{f"{m}__{k}": np.array([q[k] for q in R[m]]) for m in methods for k in R[m][0]},
         **{f"lam__{m}": np.array(LAM[m]) for m in LAM}, J_twolevel=J_twolevel, J_hist=J_hist, ytrain=y)
print(f"wall {time.time() - t_start:.0f} s")
