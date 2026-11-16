"""Lab 65_ensemble: 5-fold CV on expt_gap (Magpie, 132 features, n=4601).
(A) regression: dummy, ridge, single tree, RF, XGBoost, LightGBM, stacking (out-of-fold) vs naive stacking;
(B) classification metal (gap=0) vs non-metal: logistic, tree, AdaBoost, RF, XGBoost, LightGBM.
Same KFold(seed 0) splits for all models."""
import os, time, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb, lightgbm as lgb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]).to_numpy(np.float32)
y = df["target"].to_numpy()
print(f"data: n={len(y)}, p={X.shape[1]} Magpie features, metals (gap=0) = {np.mean(y==0):.3f}")

def reg_models():
    return {
        "ridge":    make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "tree":     DecisionTreeRegressor(min_samples_leaf=3, random_state=0),
        "RF":       RandomForestRegressor(n_estimators=200, max_features=0.33, n_jobs=2, random_state=0),
        "XGBoost":  xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                     colsample_bytree=0.5, min_child_weight=2, reg_lambda=1.0, n_jobs=1, random_state=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.5, min_child_samples=10,
                                      n_jobs=1, random_state=0, verbose=-1),
    }

t0 = time.time()
kf = KFold(5, shuffle=True, random_state=0)
names = ["dummy"] + list(reg_models()) + ["stack (OOF)", "stack (naive)"]
mae = {k: [] for k in names}; oof = {k: np.zeros(len(y)) for k in names}
wts_oof, wts_naive = [], []
base = ["ridge", "RF", "XGBoost", "LightGBM"]
for fold, (tr, te) in enumerate(kf.split(X)):
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    p = DummyRegressor().fit(Xtr, ytr).predict(Xte); oof["dummy"][te] = p
    full_te, full_tr = {}, {}
    for k, m in reg_models().items():
        m.fit(Xtr, ytr); full_te[k] = m.predict(Xte); full_tr[k] = m.predict(Xtr)
        oof[k][te] = full_te[k]
    # inner 3-fold out-of-fold predictions for the meta-learner
    Z = np.zeros((len(tr), len(base)))
    for itr, ite in KFold(3, shuffle=True, random_state=1).split(Xtr):
        ms = reg_models()
        for j, k in enumerate(base):
            Z[ite, j] = ms[k].fit(Xtr[itr], ytr[itr]).predict(Xtr[ite])
    Zte = np.column_stack([full_te[k] for k in base])
    meta = LinearRegression(positive=True).fit(Z, ytr)            # Wolpert stacking
    oof["stack (OOF)"][te] = meta.predict(Zte); wts_oof.append(np.r_[meta.intercept_, meta.coef_])
    Zin = np.column_stack([full_tr[k] for k in base])             # in-sample predictions: leaky
    metan = LinearRegression(positive=True).fit(Zin, ytr)
    oof["stack (naive)"][te] = metan.predict(Zte); wts_naive.append(np.r_[metan.intercept_, metan.coef_])
    for k in names:
        mae[k].append(np.mean(np.abs(oof[k][te] - y[te])))
    print(f"  fold {fold} done ({time.time()-t0:.0f} s)")

print("=== Part A: regression, 5-fold CV (KFold shuffle, seed 0) ===")
print(f"{'model':15s} {'MAE mean':>9s} {'+/- sd':>7s} {'RMSE':>6s}  {'MAE gap>0':>9s}")
res = {}
for k in names:
    rm = np.sqrt(np.mean((oof[k] - y) ** 2)); nz = y > 0
    res[k] = dict(mae=float(np.mean(mae[k])), sd=float(np.std(mae[k], ddof=1)), rmse=float(rm),
                  mae_nz=float(np.mean(np.abs(oof[k][nz] - y[nz]))))
    print(f"{k:15s} {np.mean(mae[k]):9.3f} {np.std(mae[k], ddof=1):7.3f} {rm:6.3f}  {res[k]['mae_nz']:9.3f}")
w1, w2 = np.mean(wts_oof, 0), np.mean(wts_naive, 0)
print("meta weights [intercept, " + ", ".join(base) + "] (mean over folds)")
print("  OOF  :", np.array2string(w1, precision=3))
print("  naive:", np.array2string(w2, precision=3))
C = np.corrcoef(np.column_stack([oof[k] - y for k in base]).T)
print("correlation of out-of-fold residuals (ridge, RF, XGB, LGBM):")
print(np.array2string(C, precision=2))
print(f"Part A time {time.time()-t0:.0f} s")

# ---------------- Part B: classification ----------------
t1 = time.time()
yc = (y > 0).astype(int)
cls = {
    "logistic":   make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "tree":       DecisionTreeClassifier(min_samples_leaf=3, random_state=0),
    "AdaBoost":   AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=300,
                                     learning_rate=0.5, random_state=0),
    "AdaBoost-d3": AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=300,
                                      learning_rate=0.5, random_state=0),
    "RF":         RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0),
    "XGBoost":    xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                    colsample_bytree=0.5, n_jobs=1, random_state=0),
    "LightGBM":   lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.5, n_jobs=1, random_state=0, verbose=-1),
}
skf = StratifiedKFold(5, shuffle=True, random_state=0)
print("=== Part B: metal (gap=0) vs non-metal, 5-fold stratified CV ===")
print(f"{'model':12s} {'accuracy':>9s} {'ROC-AUC':>8s}")
resc = {}
for k, m in cls.items():
    acc, auc = [], []
    for tr, te in skf.split(X, yc):
        m.fit(X[tr], yc[tr]); pr = m.predict_proba(X[te])[:, 1]
        acc.append(accuracy_score(yc[te], pr > 0.5)); auc.append(roc_auc_score(yc[te], pr))
    resc[k] = dict(acc=float(np.mean(acc)), acc_sd=float(np.std(acc, ddof=1)), auc=float(np.mean(auc)))
    print(f"{k:12s} {np.mean(acc):9.3f} {np.mean(auc):8.3f}   (acc sd {np.std(acc, ddof=1):.3f})")
print(f"Part B time {time.time()-t1:.0f} s; total {time.time()-t0:.0f} s")
json.dump(dict(reg=res, cls=resc, oof={k: oof[k].tolist() for k in ["XGBoost", "stack (OOF)", "tree"]},
               y=y.tolist()), open(os.path.join(HERE, "65_ensemble.json"), "w"))
