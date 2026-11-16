"""Lab 65_ens, supplement: why ridge regression's CV MAE explodes in one fold.
Same KFold(5, shuffle, seed 0) and same ridge pipeline as 65_ensemble.py."""
import os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]); y = df["target"].to_numpy()
oof = np.zeros(len(y)); zmax = np.zeros(len(y)); zfeat = np.empty(len(y), dtype=object)
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X.iloc[tr].to_numpy(np.float32), y[tr])
    oof[te] = m.predict(X.iloc[te].to_numpy(np.float32))
    Z = np.abs(m[0].transform(X.iloc[te].to_numpy(np.float32)))
    zmax[te] = Z.max(1); zfeat[te] = X.columns[Z.argmax(1)]
    print(f"fold {f}: ridge MAE = {np.mean(np.abs(oof[te]-y[te])):.3f}  max |pred| = {np.abs(oof[te]).max():.1f} eV")
w = np.argsort(-np.abs(oof - y))[:3]
for i in w:
    print(f"  {df.formula[i]:22s} true {y[i]:.2f}  ridge {oof[i]:9.1f} eV  max|z| = {zmax[i]:.0f} ({zfeat[i].replace('MagpieData ','')})")
keep = np.abs(oof - y) < 100
print(f"ridge MAE without the {np.sum(~keep)} compound(s) with |error| > 100 eV: {np.mean(np.abs(oof[keep]-y[keep])):.3f}")
print(f"ridge MAE with predictions clipped to [0, {y.max():.1f}] eV: {np.mean(np.abs(np.clip(oof, 0, y.max())-y)):.3f}")
c = "MagpieData minimum GSmagmom"
print(f"'{c.replace('MagpieData ','')}' is nonzero for {int((X[c]!=0).sum())} of {len(y)} compounds: "
      + ", ".join(f"{f} ({v:.3f})" for f, v in zip(df.formula[X[c]!=0], X[c][X[c]!=0])))
