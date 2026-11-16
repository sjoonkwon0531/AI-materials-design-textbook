"""Lab 65_boost: (A) gradient boosting as stagewise residual fitting on a 1D toy problem;
(B) check XGBoost's leaf weight w* = -G/(H+lambda) and split gain against a hand computation
on the expt_gap Magpie data (one depth-1 tree, squared loss)."""
import os, json, warnings
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- Part A: 1D toy, L2 boosting with depth-2 trees ----------------
rng = np.random.default_rng(1)
n = 200
x = np.sort(rng.uniform(0, 1, n)); f = np.sin(2 * np.pi * x) + 0.5 * (x > 0.6)
y = f + rng.normal(0, 0.25, n)
xg = np.linspace(0, 1, 501); fg = np.sin(2 * np.pi * xg) + 0.5 * (xg > 0.6)
xt = rng.uniform(0, 1, 2000); yt = np.sin(2 * np.pi * xt) + 0.5 * (xt > 0.6) + rng.normal(0, 0.25, 2000)
nu, M = 0.1, 500
F = np.full(n, y.mean()); Fg = np.full(501, y.mean()); Ft = np.full(2000, y.mean())
snap, tr_curve, te_curve = {}, [], []
for m in range(1, M + 1):
    r = y - F                                  # negative gradient of (1/2)(y-F)^2
    h = DecisionTreeRegressor(max_depth=2).fit(x[:, None], r)
    F += nu * h.predict(x[:, None]); Fg += nu * h.predict(xg[:, None]); Ft += nu * h.predict(xt[:, None])
    tr_curve.append(np.mean((y - F) ** 2)); te_curve.append(np.mean((yt - Ft) ** 2))
    if m in (1, 10, 50, 500):
        snap[m] = Fg.copy()
print("=== Part A: L2 gradient boosting, depth-2 trees, nu=0.1, n=200, noise sd=0.25 (var 0.0625) ===")
for m in (1, 10, 50, 100, 200, 500):
    print(f"  M={m:3d}: train MSE = {tr_curve[m-1]:.4f}   test MSE (n=2000) = {te_curve[m-1]:.4f}")
mbest = int(np.argmin(te_curve)) + 1
print(f"  best test MSE {min(te_curve):.4f} at M={mbest}")
json.dump(dict(x=x.tolist(), y=y.tolist(), xg=xg.tolist(), fg=fg.tolist(),
               snap={k: v.tolist() for k, v in snap.items()}, tr=tr_curve, te=te_curve),
          open(os.path.join(HERE, "65_boost_toy.json"), "w"))

# ---------------- Part B: XGBoost leaf weight and gain by hand ----------------
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]); yv = df["target"].to_numpy()
lam, base = 1.0, float(yv.mean())
bst = xgb.train(dict(objective="reg:squarederror", max_depth=1, eta=1.0, reg_lambda=lam,
                     base_score=base, tree_method="exact", min_child_weight=0),
                xgb.DMatrix(X, label=yv), num_boost_round=1)
tree = bst.trees_to_dataframe()
root = tree.iloc[0]; feat, thr = root["Feature"], root["Split"]
g = base - yv; hss = np.ones_like(yv)               # squared loss: g = yhat - y, h = 1
L = X[feat].to_numpy() < thr
GL, HL, GR, HR = g[L].sum(), hss[L].sum(), g[~L].sum(), hss[~L].sum()
wL, wR = -GL / (HL + lam), -GR / (HR + lam)
gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR)**2 / (HL + HR + lam)
leaves = tree[tree.Feature == "Leaf"]
print("=== Part B: one XGBoost stump on expt_gap (n=4601), squared loss, lambda=1, eta=1 ===")
print(f"  root split chosen by XGBoost: {feat} < {thr:.4f}")
print(f"  left : n={L.sum():4d}  G={GL:9.2f}  H={HL:6.0f}  -G/(H+lam)={wL:+.5f}   xgboost leaf={leaves.Gain.iloc[0]:+.5f}")
print(f"  right: n={(~L).sum():4d}  G={GR:9.2f}  H={HR:6.0f}  -G/(H+lam)={wR:+.5f}   xgboost leaf={leaves.Gain.iloc[1]:+.5f}")
print(f"  gain by hand GL^2/(HL+lam)+GR^2/(HR+lam)-G^2/(H+lam) = {gain:.3f}   xgboost root gain = {root['Gain']:.3f}")
