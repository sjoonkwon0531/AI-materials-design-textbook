"""Lab 65_lc: learning curves (test MAE vs training-set size) on expt_gap (Magpie).
5-fold KFold(seed 0); within each training fold, nested subsets of size n are drawn (seed = fold)."""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]).to_numpy(np.float32); y = df["target"].to_numpy()
mk = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "tree": lambda: DecisionTreeRegressor(min_samples_leaf=3, random_state=0),
    "RF": lambda: RandomForestRegressor(n_estimators=150, max_features=0.33, n_jobs=2, random_state=0),
    "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                          subsample_freq=1, colsample_bytree=0.5, min_child_samples=10,
                                          n_jobs=1, random_state=0, verbose=-1),
}
sizes = [250, 500, 1000, 2000, 3680]
t0 = time.time()
te_mae = {k: np.zeros((5, len(sizes))) for k in mk}; tr_mae = {k: np.zeros((5, len(sizes))) for k in mk}
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
    order = np.random.default_rng(f).permutation(tr)
    for j, n in enumerate(sizes):
        sub = order[:n]
        for k, g in mk.items():
            m = g().fit(X[sub], y[sub])
            te_mae[k][f, j] = np.mean(np.abs(m.predict(X[te]) - y[te]))
            tr_mae[k][f, j] = np.mean(np.abs(m.predict(X[sub]) - y[sub]))
print("=== learning curves: test MAE (eV), mean +/- sd over 5 folds; train MAE in brackets ===")
print("n_train  " + "  ".join(f"{k:>20s}" for k in mk))
for j, n in enumerate(sizes):
    print(f"{n:7d}  " + "  ".join(f"{te_mae[k][:, j].mean():6.3f}+/-{te_mae[k][:, j].std(ddof=1):.3f} [{tr_mae[k][:, j].mean():.3f}]" for k in mk))
for k in ["RF", "LightGBM"]:
    s = np.polyfit(np.log(sizes[1:]), np.log(te_mae[k].mean(0)[1:]), 1)[0]
    print(f"{k}: log-log slope of test MAE vs n (n>=500) = {s:.3f}")
print(f"time {time.time()-t0:.0f} s")
json.dump(dict(sizes=sizes, te={k: v.tolist() for k, v in te_mae.items()}, tr={k: v.tolist() for k, v in tr_mae.items()}),
          open(os.path.join(HERE, "65_lc.json"), "w"))
