"""Lab 7.6(a): an end-to-end PyTorch MLP band-gap predictor on the Ch.6 feature table (149 features),
evaluated on the SAME splits as Ch.6 Lab 6.16 (random 5-fold seed 0; GroupKFold(5) by chemical system),
compared with LightGBM rerun with the Ch.6 settings. Ensemble of K MLPs per fold -> predictive spread.
Run: python 76_mlp_bandgap.py  (about 5-7 min on one shared CPU thread)."""
import os, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
t_start = time.time()

df = pd.read_parquet(os.path.join(DATA, "67_features_clean.parquet"))
feat = [c for c in df.columns if c not in ("formula", "target", "chemsys", "reduced")]
X = df[feat].values.astype(np.float32); y = df.target.values.astype(np.float32)
print("n = %d, p = %d, metals %.1f%%" % (len(y), X.shape[1], 100 * (y == 0).mean()))

class GapData(Dataset):                                   # a minimal Dataset: (x, y) pairs as tensors
    def __init__(self, X, y): self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_mlp(p, h=256, depth=3, drop=0.1):
    layers, d = [], p
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(drop)]; d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)

def fit_one(Xtr, ytr, Xva, yva, seed, epochs=80, bs=128, lr=2e-3, wd=1e-4, patience=15):
    torch.manual_seed(seed)
    model = make_mlp(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(ytr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")       # linear warm-up + cosine decay
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(GapData(Xtr, ytr), batch_size=bs, shuffle=True, generator=g)
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    best, best_state, wait, ep_best = np.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            loss = (model(xb).squeeze(-1) - yb).abs().mean()           # L1 loss = the reported metric
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            v = (model(Xv).squeeze(-1).clamp_min(0) - yv).abs().mean().item()
        if v < best - 1e-4: best, best_state, wait, ep_best = v, {k: t.clone() for k, t in model.state_dict().items()}, 0, ep
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); model.eval()
    return model, ep_best, ep

def prep(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)                         # fitted on the training part ONLY
    f = lambda A: np.clip(sc.transform(A), -6, 6).astype(np.float32)   # tame near-constant columns
    return f(Xtr), f(Xte)

P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=1, verbose=-1)   # = Ch.6 Lab 6.16
K = 4
schemes = {"random 5-fold (seed 0)": list(KFold(5, shuffle=True, random_state=0).split(X)),
           "group 5-fold (chemsys)": list(GroupKFold(5).split(X, y, df.chemsys.values))}
out = {}
for name, splits in schemes.items():
    t0 = time.time()
    o_gb = np.zeros_like(y); o_mu = np.zeros_like(y); o_sd = np.zeros_like(y); o_single = np.zeros_like(y)
    epochs_used = []
    for f, (tr, te) in enumerate(splits):
        o_gb[te] = np.clip(lgb.LGBMRegressor(random_state=0, **P).fit(X[tr], y[tr]).predict(X[te]), 0, None)
        preds = []
        for k in range(K):
            itr, iva = train_test_split(tr, test_size=0.1, random_state=100 * f + k)   # inner validation
            Xa, Xb = prep(X[itr], np.vstack([X[iva], X[te]]))
            m, eb, el = fit_one(Xa, y[itr], Xb[:len(iva)], y[iva], seed=100 * f + k)
            epochs_used.append(eb + 1)
            with torch.no_grad():
                preds.append(m(torch.from_numpy(Xb[len(iva):])).squeeze(-1).clamp_min(0).numpy())
        preds = np.array(preds)
        o_single[te] = preds[0]; o_mu[te] = preds.mean(0); o_sd[te] = preds.std(0, ddof=1)
        print("  [%s] fold %d: LGBM %.3f | MLP single %.3f | MLP ens(%d) %.3f eV  (%.0f s)" % (
            name[:6], f, mean_absolute_error(y[te], o_gb[te]), mean_absolute_error(y[te], o_single[te]), K,
            mean_absolute_error(y[te], o_mu[te]), time.time() - t0))
    err = np.abs(o_mu - y)
    nm = y > 0
    r = np.corrcoef(o_sd, err)[0, 1]
    q = np.quantile(o_sd, [0.2, 0.8])
    print("%s:" % name)
    print("   LightGBM (Ch.6 settings)  MAE %.3f eV  | non-metals %.3f" % (mean_absolute_error(y, o_gb), mean_absolute_error(y[nm], o_gb[nm])))
    print("   MLP single (seed k=0)     MAE %.3f eV  | non-metals %.3f" % (mean_absolute_error(y, o_single), mean_absolute_error(y[nm], o_single[nm])))
    print("   MLP ensemble of %d        MAE %.3f eV  | non-metals %.3f" % (K, mean_absolute_error(y, o_mu), mean_absolute_error(y[nm], o_mu[nm])))
    print("   average of MLP-ens and LGBM MAE %.3f eV" % mean_absolute_error(y, 0.5 * (o_mu + o_gb)))
    print("   early-stopped epoch: median %d (range %d-%d)" % (np.median(epochs_used), min(epochs_used), max(epochs_used)))
    print("   ensemble spread vs |error|: Pearson r = %.2f; MAE in lowest/highest spread quintile %.3f / %.3f eV" % (
        r, err[o_sd <= q[0]].mean(), err[o_sd >= q[1]].mean()))
    print("   coverage of mu +/- 1.96 sd: %.1f%% (nominal 95%%)" % (100 * (err <= 1.96 * o_sd).mean()))
    out[name] = dict(y=y.tolist(), gb=o_gb.tolist(), mu=o_mu.tolist(), sd=o_sd.tolist(), single=o_single.tolist())
json.dump(out, open(os.path.join(HERE, "76_mlp_bandgap_oof.json"), "w"))
print("wall %.0f s (single thread, shared machine)" % (time.time() - t_start))
