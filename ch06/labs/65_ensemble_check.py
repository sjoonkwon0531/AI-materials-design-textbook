"""Lab 6.13 companion: out-of-fold residual correlations of 65_ensemble.py (same models, same
KFold(5, shuffle, seed 0)), computed on all 4,601 compounds and again without Fe9Co7, whose
36,025 eV ridge error dominates the ridge residual vector (see 65_ridge_outlier.py)."""
import os, time, warnings, importlib.util
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); t0 = time.time()
src = open(os.path.join(HERE, "65_ensemble.py")).read()
ns = {"__file__": os.path.join(HERE, "65_ensemble.py")}
exec(src.split("t0 = time.time()")[0], ns)                 # imports, data and reg_models() only
X, y, df, reg_models = ns["X"], ns["y"], ns["df"], ns["reg_models"]
base = ["ridge", "RF", "XGBoost", "LightGBM"]
oof = {k: np.zeros(len(y)) for k in base}
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    ms = reg_models()
    for k in base:
        oof[k][te] = ms[k].fit(X[tr], y[tr]).predict(X[te])
R = np.column_stack([oof[k] - y for k in base])
keep = (df.formula != "Fe9Co7").to_numpy()
np.set_printoptions(precision=3, suppress=True)
print("correlation of out-of-fold residuals (ridge, RF, XGB, LGBM), all %d compounds:" % len(y))
print(np.corrcoef(R.T))
print("same, without Fe9Co7 (n = %d):" % keep.sum())
print(np.corrcoef(R[keep].T))
print("ridge MAE without Fe9Co7 = %.3f eV" % np.mean(np.abs(R[keep, 0])))
print("wall %.0f s" % (time.time() - t0))
