"""Lab 65_importance: impurity (MDI) vs permutation importance for an RF on expt_gap (Magpie),
with two probe columns: a pure-noise continuous feature and a near-duplicate of a strong feature."""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
X = df.drop(columns=["formula", "target"]); y = df["target"].to_numpy()
X.columns = [c.replace("MagpieData ", "") for c in X.columns]
rng = np.random.default_rng(0)
X["PROBE noise"] = rng.standard_normal(len(y))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

def fit_rank(Xtr, Xte, tag):
    rf = RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=1, n_jobs=2, random_state=0)
    rf.fit(Xtr, ytr)
    mdi = pd.Series(rf.feature_importances_, index=Xtr.columns)
    pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=1,
                                scoring="neg_mean_absolute_error")
    perm = pd.Series(pi.importances_mean, index=Xtr.columns)
    print(f"--- {tag}: test MAE = {np.mean(np.abs(rf.predict(Xte)-yte)):.3f} eV ---")
    return mdi, perm

mdi, perm = fit_rank(Xtr, Xte, "RF with noise probe")
r_mdi = mdi.rank(ascending=False); r_perm = perm.rank(ascending=False)
print("top 8 by MDI (impurity)            | top 8 by permutation (test MAE increase, eV)")
a, b = mdi.sort_values(ascending=False).head(8), perm.sort_values(ascending=False).head(8)
for (ka, va), (kb, vb) in zip(a.items(), b.items()):
    print(f"  {ka:30s} {va:.3f}   | {kb:30s} {vb:.3f}")
print(f"noise probe: MDI = {mdi['PROBE noise']:.4f} (rank {int(r_mdi['PROBE noise'])}/{len(mdi)}), "
      f"permutation = {perm['PROBE noise']:+.4f} eV (rank {int(r_perm['PROBE noise'])})")
nbelow = int((mdi < mdi['PROBE noise']).sum())
print(f"  number of real Magpie features with MDI below the pure-noise probe: {nbelow}")
# correlated features: how many Magpie features have |r| > 0.9 with the top permutation feature
top = perm.idxmax()
corr = Xtr.drop(columns=["PROBE noise"]).corrwith(Xtr[top]).abs().sort_values(ascending=False)
print(f"features with |r|>0.9 with '{top}': {int((corr>0.9).sum())-1}")

# near-duplicate of the top feature: importance gets split
X2tr, X2te = Xtr.copy(), Xte.copy()
s = Xtr[top].std()
X2tr["DUP " + top] = Xtr[top] + 0.01 * s * rng.standard_normal(len(Xtr))
X2te["DUP " + top] = Xte[top] + 0.01 * s * rng.standard_normal(len(Xte))
mdi2, perm2 = fit_rank(X2tr, X2te, "RF with near-duplicate of the top feature")
print(f"  '{top}': MDI {mdi[top]:.3f} -> {mdi2[top]:.3f} (+ duplicate {mdi2['DUP '+top]:.3f}); "
      f"permutation {perm[top]:.3f} -> {perm2[top]:.3f} (duplicate {perm2['DUP '+top]:.3f}) eV")
json.dump(dict(mdi=mdi.to_dict(), perm=perm.to_dict(), top=top), open(os.path.join(HERE, "65_importance.json"), "w"))
