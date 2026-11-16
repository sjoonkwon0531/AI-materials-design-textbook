"""Lab 65_bv: variance of averaged trees, Var = rho*sigma^2 + (1-rho)*sigma^2/B,
and bias^2/variance numbers for tree, bagging, RF, XGBoost, LightGBM on expt_gap (Magpie).
R replicate training sets (n_train each) drawn without replacement from a pool;
a fixed test set of n_test compounds. Seeds fixed. ~3-4 min on 2 cores."""
import os, time, warnings, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb, lightgbm as lgb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]).to_numpy(np.float32)
y = df["target"].to_numpy()
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
n_test, n_train, R, B = 400, 1500, 16, 100
te, pool = perm[:n_test], perm[n_test:]
Xte, yte = X[te], y[te]
t0 = time.time()

# ---------- Part A: variance vs number of trees B, for three max_features ----------
print(f"=== Part A: averaged trees, n_train={n_train}, R={R} replicate training sets, B={B}, n_test={n_test} ===")
Bgrid = [1, 2, 5, 10, 20, 50, 100]
curves = {}
for mf, name in [(1.0, "bagging (m=p)"), (0.33, "RF m=p/3"), (0.1, "RF m=p/10")]:
    P = np.empty((R, B, n_test))
    for r in range(R):
        tr = np.random.default_rng(100 + r).choice(pool, n_train, replace=False)
        rf = RandomForestRegressor(n_estimators=B, max_features=mf, n_jobs=2, random_state=r)
        rf.fit(X[tr], y[tr])
        for b, est in enumerate(rf.estimators_):
            P[r, b] = est.predict(Xte)
    mu = P.mean(axis=(0, 1))                       # E over data and tree randomness
    D = P - mu
    sigma2 = (D**2).mean(axis=(0, 1))              # single-tree variance per test point
    S = D.sum(axis=1)                              # (R, n_test)
    cov = ((S**2 - (D**2).sum(axis=1)) / (B * (B - 1))).mean(axis=0)
    s2, c = sigma2.mean(), cov.mean()              # averaged over test points
    rho = c / s2                                   # effective (sigma^2-weighted) correlation
    emp, pred = [], []
    for Bk in Bgrid:
        m = P[:, :Bk].mean(axis=1)                 # forest of the first Bk trees
        emp.append(((m - mu) ** 2).mean(axis=0).mean())
        pred.append(rho * s2 + (1 - rho) * s2 / Bk)
    curves[name] = dict(B=Bgrid, emp=emp, pred=pred, rho=float(rho), sigma2=float(s2))
    bias2 = ((yte - P.mean(axis=(0, 1))) ** 2).mean()
    mae = np.abs(P.mean(axis=1) - yte).mean()
    print(f"{name:14s} mean sigma^2 = {sigma2.mean():.3f}  rho(sigma^2-weighted) = {curves[name]['rho']:.3f}  "
          f"bias^2+noise = {bias2:.3f}  test MAE (B={B}) = {mae:.3f} eV")
    for Bk, e, p in zip(Bgrid, emp, pred):
        print(f"    B={Bk:3d}: Var(empirical) = {e:.4f}   rho*s2+(1-rho)*s2/B = {p:.4f}")
print(f"Part A time {time.time()-t0:.1f} s")

# ---------- Part B: bias^2 / variance for five learners ----------
t1 = time.time()
print(f"=== Part B: bias^2(+noise) / variance over R={R} training sets of n={n_train} ===")
models = {
    "tree (full depth)": lambda s: DecisionTreeRegressor(random_state=s),
    "tree (depth 4)":    lambda s: DecisionTreeRegressor(max_depth=4, random_state=s),
    "bagging B=100":     lambda s: RandomForestRegressor(100, max_features=1.0, n_jobs=2, random_state=s),
    "RF B=100, m=p/3":   lambda s: RandomForestRegressor(100, max_features=0.33, n_jobs=2, random_state=s),
    "XGBoost":           lambda s: xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                                    subsample=0.8, colsample_bytree=0.5, n_jobs=1, random_state=s),
    "LightGBM":          lambda s: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                                     n_jobs=1, random_state=s, verbose=-1),
}
bv = {}
print(f"{'model':20s} {'test MSE':>9s} {'bias2+noise':>12s} {'variance':>9s} {'MAE':>7s}")
for name, mk in models.items():
    P = np.empty((R, n_test))
    for r in range(R):
        tr = np.random.default_rng(100 + r).choice(pool, n_train, replace=False)
        P[r] = mk(r).fit(X[tr], y[tr]).predict(Xte)
    Fbar = P.mean(axis=0)
    mse = ((P - yte) ** 2).mean()
    b2 = ((Fbar - yte) ** 2).mean()
    var = ((P - Fbar) ** 2).mean()
    mae = np.abs(P - yte).mean()
    bv[name] = dict(mse=mse, bias2=b2, var=var, mae=mae)
    print(f"{name:20s} {mse:9.3f} {b2:12.3f} {var:9.3f} {mae:7.3f}   (check: b2+var-mse = {b2+var-mse:+.1e})")
print(f"Part B time {time.time()-t1:.1f} s; total {time.time()-t0:.1f} s")
json.dump(dict(curves=curves, bv=bv), open(os.path.join(HERE, "65_bv.json"), "w"), indent=1)
