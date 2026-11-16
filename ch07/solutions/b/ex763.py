"""Ex 7.6.3: swap the losses. MLP trained with MSE (ensemble K=2, otherwise Lab 7.18 settings; early stopping still on
validation MAE), LightGBM with objective='l1' (otherwise Ch.6 settings). Random 5-fold only."""
import time, json, numpy as np, torch
src = open("../ch07/labs/76_mlp_bandgap.py").read()
pre_src = src.split("P = dict(")[0].replace("loss = (model(xb).squeeze(-1) - yb).abs().mean()", "loss = (model(xb).squeeze(-1) - yb).pow(2).mean()")
assert ".pow(2).mean()" in pre_src
pre_src = pre_src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", "HERE = \"../ch07/labs\"")
exec(pre_src)
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=1, verbose=-1)
splits = list(KFold(5, shuffle=True, random_state=0).split(X))
o_l1 = np.zeros_like(y); o_mse1 = np.zeros_like(y); o_mse2 = np.zeros_like(y); t0 = time.time()
for f, (tr, te) in enumerate(splits):
    o_l1[te] = np.clip(lgb.LGBMRegressor(random_state=0, objective="l1", **P).fit(X[tr], y[tr]).predict(X[te]), 0, None)
    preds = []
    for k in range(2):
        itr, iva = train_test_split(tr, test_size=0.1, random_state=100 * f + k)
        Xa, Xb = prep(X[itr], np.vstack([X[iva], X[te]]))
        m, eb, el = fit_one(Xa, y[itr], Xb[:len(iva)], y[iva], seed=100 * f + k)
        with torch.no_grad(): preds.append(m(torch.from_numpy(Xb[len(iva):])).squeeze(-1).clamp_min(0).numpy())
    o_mse1[te] = preds[0]; o_mse2[te] = np.mean(preds, 0)
    print(f"fold {f}: LGBM-l1 {np.abs(o_l1[te]-y[te]).mean():.3f} | MLP-MSE single {np.abs(o_mse1[te]-y[te]).mean():.3f} ens2 {np.abs(o_mse2[te]-y[te]).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
d = json.load(open("../ch07/labs/76_mlp_bandgap_oof.json"))["random 5-fold (seed 0)"]
met = y == 0
def row(name, p):
    p = np.asarray(p); e = np.abs(p - y)
    print(f"{name:26s} all {e.mean():.3f} | metals {e[met].mean():.3f} | non-metals {e[~met].mean():.3f} | "
          f"frac of metals predicted < 0.01 eV {(p[met] < 0.01).mean():.2f} | median pred on metals {np.median(p[met]):.3f}")
row("LightGBM L2 (Lab 7.18)", d["gb"]); row("LightGBM L1 (new)", o_l1)
row("MLP L1 single (Lab 7.18)", d["single"]); row("MLP L1 ens4 (Lab 7.18)", d["mu"])
row("MLP MSE single (new)", o_mse1); row("MLP MSE ens2 (new)", o_mse2)
json.dump(dict(l1gb=o_l1.tolist(), mse1=o_mse1.tolist(), mse2=o_mse2.tolist()), open("ex763.json", "w"))
print(f"wall {time.time()-t0:.0f} s")
