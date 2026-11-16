"""Lab 8.3 (a): AdamW vs SAM vs ASAM vs F-SAM (and a large-batch AdamW) on the Ch.6/7 band-gap MLP.
In-distribution: random 80/20 split of non-oxide, non-halide compounds.  OOD: all oxides, all halides
(families as in Ch.6 Lab 6.16: most electronegative element).  Measures test MAE, Hessian lambda_max of the
training MSE (Lanczos on HVPs, float64), the scale-normalised lambda_max of T H T (T = |w|), and saves models
for Lab 8.3 (b).  Usage: python 83_bandgap_sam.py   (~6-10 min on one thread under load)"""
import os, sys, copy, time, json, importlib.util
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("sam83", os.path.join(HERE, "83_sam.py"))
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
t0 = time.time()

df = pd.read_parquet(os.path.join(HERE, "..", "..", "ch07", "data", "67_features_clean.parquet"))
feat = [c for c in df.columns if c not in ("formula", "target", "chemsys", "reduced")]
X, y = df[feat].values.astype(np.float64), df.target.values.astype(np.float32)
FAM = {**{e: "halide" for e in ("F", "Cl", "Br", "I")}, "O": "oxide",
       **{e: "chalcogenide" for e in ("S", "Se", "Te")}, **{e: "pnictide" for e in ("N", "P", "As", "Sb")}}
fam = np.array([FAM.get(max(Composition(f).elements, key=lambda e: e.X if e.X == e.X else 0).symbol, "other")
                for f in df.formula])
ood_ox, ood_ha = np.where(fam == "oxide")[0], np.where(fam == "halide")[0]
pool = np.where((fam != "oxide") & (fam != "halide"))[0]
perm = np.random.default_rng(0).permutation(pool)
n_te = len(pool) // 5
te, tr_all = perm[:n_te], perm[n_te:]
va, tr = tr_all[:len(tr_all) // 8], tr_all[len(tr_all) // 8:]       # validation only for choosing rho
sc = StandardScaler().fit(X[tr_all])
Z = lambda idx: torch.from_numpy(np.clip(sc.transform(X[idx]), -6, 6).astype(np.float32))
Y = lambda idx: torch.from_numpy(y[idx])
print(f"n: train {len(tr_all)} (of which val {len(va)}), ID test {len(te)}, OOD oxide {len(ood_ox)}, halide {len(ood_ha)}")
print("mean gap: train %.3f  ID %.3f  oxide %.3f  halide %.3f eV; metals in train %.0f%%" % (
    y[tr_all].mean(), y[te].mean(), y[ood_ox].mean(), y[ood_ha].mean(), 100 * (y[tr_all] == 0).mean()))


def make_mlp(p, h=256, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))


def fit(Xt, yt, method, seed, rho=0.05, epochs=60, bs=64, lr=1e-3, wd=1e-4, noise=0.0, fgsm=0.0):
    model = make_mlp(Xt.shape[1], seed=seed)
    kw = dict(lr=lr, weight_decay=wd)
    if method in ("sam", "asam", "fsam"): opt = S.SAM(model.parameters(), torch.optim.AdamW, rho=rho, mode=method, **kw)
    else: opt = torch.optim.AdamW(model.parameters(), **kw)
    gen = torch.Generator().manual_seed(seed); n = len(yt)
    nb = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base if hasattr(opt, "base") else opt, epochs * nb)
    for ep in range(epochs):
        idx = torch.randperm(n, generator=gen)
        for b in range(nb):
            j = idx[b * bs:(b + 1) * bs]; xb, yb = Xt[j], yt[j]
            if noise > 0: xb = xb + noise * torch.randn(xb.shape, generator=gen)      # input-noise augmentation
            if fgsm > 0:                                                                # FGSM adversarial training
                xa = xb.clone().requires_grad_(True)
                gx, = torch.autograd.grad((model(xa).squeeze(-1) - yb).pow(2).mean(), xa)
                xb = torch.cat([xb, (xb + fgsm * gx.sign()).detach()]); yb = torch.cat([yb, yb])
            def closure():
                opt.zero_grad(); loss = (model(xb).squeeze(-1) - yb).pow(2).mean(); loss.backward(); return loss
            if hasattr(opt, "base"): opt.step(closure)
            else: closure(); opt.step()
            sched.step()
    return model


@torch.no_grad()
def mae(model, idx): return (model(Z(idx)).squeeze(-1).clamp(min=0) - Y(idx)).abs().mean().item()


def sharpness(model, idx, adaptive=False):
    model = copy.deepcopy(model).double()                           # float64 copy for a converged eigsh estimate
    Xs, ys = Z(idx).double(), Y(idx).double()
    ps = list(model.parameters())
    if not adaptive:
        return S.lambda_max(lambda: (model(Xs).squeeze(-1) - ys).pow(2).mean(), ps)[0]
    # lambda_max of T H T with T = |w|: Hessian in the scale-normalised coordinates u = w / |w0|
    w0 = [p.detach().clone() for p in ps]; T = [w.abs() for w in w0]
    u = [torch.ones_like(w).requires_grad_(True) for w in w0]
    def loss_u():
        params = {n: ui * w for (n, _), ui, w in zip(model.named_parameters(), u, w0)}
        out = torch.func.functional_call(model, params, (Xs,))
        return (out.squeeze(-1) - ys).pow(2).mean()
    return S.lambda_max(loss_u, u)[0]


sub = np.random.default_rng(1).choice(tr_all, 1024, replace=False)     # fixed subset for sharpness
res = {}
# ---- (1) choose rho on the validation split (seed 0) -------------------------------------------------------
Xtr, ytr = Z(tr), Y(tr)
RHO = {"sam": (0.02, 0.05, 0.2), "asam": (0.2, 0.5, 2.0), "fsam": (0.02, 0.05, 0.2)}
best_rho = {}
for m, grid in RHO.items():
    scores = []
    for r in grid:
        mod = fit(Xtr, ytr, m, seed=0, rho=r); scores.append(mae(mod, va))
    best_rho[m] = grid[int(np.argmin(scores))]
    print(f"  rho selection {m:5s}: " + "  ".join(f"rho={r:g} val MAE {s:.3f}" for r, s in zip(grid, scores))
          + f"  -> {best_rho[m]:g}   ({time.time() - t0:.0f} s)", flush=True)
# ---- (2) retrain on train+val with 5 seeds ----------------------------------------------------------------
Xtr, ytr = Z(tr_all), Y(tr_all)
METHODS = [("AdamW (bs 64)", "adamw", dict()), ("AdamW (bs 1024)", "adamw", dict(bs=1024, lr=4e-3)),
           ("SAM", "sam", dict(rho=best_rho["sam"])), ("ASAM", "asam", dict(rho=best_rho["asam"])),
           ("F-SAM", "fsam", dict(rho=best_rho["fsam"])),
           ("AdamW + input noise", "adamw", dict(noise=0.1)), ("AdamW + FGSM", "adamw", dict(fgsm=0.05))]
for name, m, kw in METHODS:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        c0 = time.process_time()
        mod = fit(Xtr, ytr, m, seed=seed, **kw)
        r = dict(id=mae(mod, te), ox=mae(mod, ood_ox), ha=mae(mod, ood_ha), train=mae(mod, tr_all),
                 lam=sharpness(mod, sub), lam_ad=sharpness(mod, sub, adaptive=True), cpu=time.process_time() - c0)
        rows.append(r)
        torch.save(mod.state_dict(), os.path.join(HERE, f"83_bg_{m}{'_' + list(kw)[0] if m == 'adamw' and kw else ''}_s{seed}.pt"))
    res[name] = rows
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print("%-20s train %.3f | ID %.3f+/-%.3f | oxide %.3f+/-%.3f | halide %.3f+/-%.3f | lam_max %7.1f+/-%5.1f | "
          "lam(THT) %6.2f+/-%5.2f | cpu %3.0f s/model" % (name, A["train"].mean(), A["id"].mean(), A["id"].std(),
          A["ox"].mean(), A["ox"].std(), A["ha"].mean(), A["ha"].std(), A["lam"].mean(), A["lam"].std(),
          A["lam_ad"].mean(), A["lam_ad"].std(), A["cpu"].mean()), flush=True)
json.dump(dict(res=res, best_rho=best_rho), open(os.path.join(HERE, "83_bandgap_results.json"), "w"))
np.savez(os.path.join(HERE, "83_bandgap_split.npz"), tr_all=tr_all, te=te, ox=ood_ox, ha=ood_ha, sub=sub)
print("wall %.0f s" % (time.time() - t0))
