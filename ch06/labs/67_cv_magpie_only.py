"""Lab 6.16 companion: the same LightGBM settings and random 5-fold split (seed 0) as 67_cv_leakage.py,
run on the 132 Magpie features only and on all 149 features, with raw and clipped (>= 0) predictions.
Separates the effect of the 17 extra features from the effect of clipping negative predictions at 0."""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); t0 = time.time()
df = pd.read_parquet(os.path.join(HERE, "67_features_clean.parquet"))
feat = [c for c in df.columns if c not in ("formula", "target", "chemsys", "reduced")]
mag = [c for c in feat if c.startswith("MagpieData")]
y = df.target.values
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=1, verbose=-1)   # = 67_cv_leakage.py
splits = list(KFold(5, shuffle=True, random_state=0).split(df))
print("n = %d; random 5-fold (seed 0); LightGBM settings as in 67_cv_leakage.py" % len(y))
for name, cols in [("Magpie only", mag), ("all features", feat)]:
    X = df[cols].values; raw = np.zeros_like(y)
    for tr, te in splits:
        raw[te] = lgb.LGBMRegressor(random_state=0, **P).fit(X[tr], y[tr]).predict(X[te])
    print("%-13s (p = %3d): MAE raw = %.3f eV, clipped at 0 = %.3f eV" % (
        name, len(cols), mean_absolute_error(y, raw), mean_absolute_error(y, np.clip(raw, 0, None))))
print("wall %.0f s" % (time.time() - t0))
