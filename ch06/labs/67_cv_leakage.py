"""Lab 6.7 step 2: random K-fold vs grouped (chemical system) vs leave-cluster-out CV.
Direct LightGBM regression of the experimental band gap on 149 composition features."""
import os, time, warnings, json
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); t0 = time.time()
df = pd.read_parquet(os.path.join(HERE, "67_features_clean.parquet"))
feat = [c for c in df.columns if c not in ("formula", "target", "chemsys", "reduced")]
X, y = df[feat].values, df.target.values
C = np.corrcoef(StandardScaler().fit_transform(X).T)
iu = np.triu_indices_from(C, 1)
print("feature pairs with |r|>0.95: %d of %d" % ((np.abs(C[iu]) > 0.95).sum(), len(iu[0])))
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=1, verbose=-1)
def run(splits, name):
    oof = np.zeros_like(y); fold_mae = []
    for tr, te in splits:
        m = lgb.LGBMRegressor(random_state=0, **P).fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, None); fold_mae.append(mean_absolute_error(y[te], oof[te]))
    print("%-34s MAE = %.3f eV  (fold mean %.3f +/- %.3f)" % (name, mean_absolute_error(y, oof),
          np.mean(fold_mae), np.std(fold_mae)))
    return oof, fold_mae
from pymatgen.core import Composition
FAM = {**{e: "halide" for e in ("F", "Cl", "Br", "I")}, "O": "oxide",
       **{e: "chalcogenide" for e in ("S", "Se", "Te")}, **{e: "pnictide" for e in ("N", "P", "As", "Sb")}}
fam = np.array([FAM.get(max(Composition(f).elements, key=lambda e: e.X if e.X == e.X else 0).symbol, "other")
                for f in df.formula])
print("families:", pd.Series(fam).value_counts().to_dict())
res = {}
print("baseline: predict training mean -> MAE = %.3f eV" % mean_absolute_error(y, np.full_like(y, y.mean())))
for seed in (0, 1, 2):
    o, f = run(KFold(5, shuffle=True, random_state=seed).split(X), f"random 5-fold (seed {seed})")
    res.setdefault("random", []).append(f)
o_rand = o
grp = df.chemsys.values
o_grp, f = run(GroupKFold(5).split(X, y, grp), "group 5-fold (chemical system)")
res["group"] = f
# a stricter group: the element set with oxidation-insensitive anion family -> "element-containing" clusters
Z = StandardScaler().fit_transform(X)
for k in (5, 10):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    o, f = run(((np.where(lab != c)[0], np.where(lab == c)[0]) for c in range(k)), f"leave-one-cluster-out (k={k})")
    for c in range(k):
        m = lab == c; fc = pd.Series(fam[m]).value_counts(normalize=True)
        print("    cluster %d: n=%4d  mean gap %.2f eV  metal %.0f%%  top family %-12s (%.0f%%)  MAE %.2f eV" % (
            c, m.sum(), y[m].mean(), 100 * (y[m] == 0).mean(), fc.index[0], 100 * fc.iloc[0], f[c]))
    res[f"loco{k}"] = f; res[f"lab{k}"] = lab.tolist(); res["fam"] = fam.tolist()
np.save(os.path.join(HERE, "67_oof_random.npy"), o_rand); np.save(os.path.join(HERE, "67_oof_group.npy"), o_grp)
json.dump(res, open(os.path.join(HERE, "67_cv_results.json"), "w"))
print("wall %.0f s" % (time.time() - t0))
