"""Lab 6.4.1 — PCA of the Magpie feature space of matbench_expt_gap (all 4,601
compositions). Scaling matters, explained variance, loadings, and principal-
component regression (PCR) as truncated-SVD regularization. ~30 s on 2 cores."""
import os, time, warnings
import numpy as np, pandas as pd
from pymatgen.core import Composition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
d = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
feats = [c for c in d.columns if c.startswith("MagpieData")]
X = d[feats].to_numpy(float); y = d.target.to_numpy(float)
names = np.array([f.replace("MagpieData ", "") for f in feats])
keep = X.std(0) > 0
frac_mode = np.array([pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(X.shape[1])])
keep &= frac_mode <= 0.99
X, names = X[:, keep], names[keep]
n, p = X.shape
metal = (y == 0)
print(f"all compositions: n = {n}, p = {p} (constant / >99%-identical columns removed); metals: {metal.sum()}")

print("\n=== Part A: raw vs standardized ===")
pr = PCA().fit(X)
j = np.argmax(np.abs(pr.components_[0]))
print(f"raw features: PC1 explains {pr.explained_variance_ratio_[0]:.3f} of variance; "
      f"largest |loading| = {names[j]} ({pr.components_[0, j]:+.3f})")
sd = X.std(0); o = np.argsort(-sd)[:3]
print("three largest raw standard deviations: " + ", ".join(f"{names[k]} ({sd[k]:.0f})" for k in o))
Z = StandardScaler().fit_transform(X)
ps = PCA().fit(Z)
evr = ps.explained_variance_ratio_; cum = np.cumsum(evr)
print("standardized: EVR of PC1..PC5 = " + ", ".join(f"{v:.3f}" for v in evr[:5]))
for t in [0.8, 0.9, 0.95, 0.99]:
    print(f"   components for {int(t*100)}% variance: {np.searchsorted(cum, t) + 1}")
# PCA = SVD of the centered data (Recall Ch.1 §1.5.8)
U, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
print(f"check: max |sigma_i^2/(n-1) - eigenvalue_i| = {np.abs(s**2/(n-1) - ps.explained_variance_).max():.2e}; "
      f"|<v1_svd, v1_pca>| = {abs(Vt[0] @ ps.components_[0]):.12f}")

print("\n=== Part B: loadings ===")
for k in range(3):
    o = np.argsort(-np.abs(ps.components_[k]))[:5]
    print(f"PC{k+1}: " + ", ".join(f"{names[i]} ({ps.components_[k, i]:+.2f})" for i in o))
T = ps.transform(Z)[:, :2]
for k in range(2):
    print(f"corr(PC{k+1} score, band gap) = {np.corrcoef(T[:, k], y)[0, 1]:+.3f}")
print(f"PC1 score: metals mean {T[metal, 0].mean():+.2f}, non-metals mean {T[~metal, 0].mean():+.2f}")

print("\n=== Part C: PCA as a regularizer (PCR) on the non-metals, grouped 5-fold CV ===")
nm = ~metal; Xn, yn = X[nm], y[nm]
fm = np.array([pd.Series(Xn[:, j]).value_counts(normalize=True).iloc[0] for j in range(p)])
Xn = Xn[:, (fm <= 0.99) & (Xn.std(0) > 0)]
print(f"non-metals: n = {len(yn)}, p = {Xn.shape[1]} after re-applying the >99%-identical filter to this subset")
chemsys = np.array(["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in d.formula[nm]])
cv = GroupKFold(5)
for k in [2, 5, 10, 20, 40, 80, 120]:
    m = make_pipeline(StandardScaler(), PCA(k), LinearRegression())
    pr_ = cross_val_predict(m, Xn, yn, cv=cv, groups=chemsys)
    print(f"   PCR k = {k:3d}: MAE = {np.mean(np.abs(pr_ - yn)):.3f} eV")
m = make_pipeline(StandardScaler(), Ridge(alpha=31.6))
pr_ = cross_val_predict(m, Xn, yn, cv=cv, groups=chemsys)
print(f"   ridge (alpha = 31.6): MAE = {np.mean(np.abs(pr_ - yn)):.3f} eV")

print("\n=== Part D: metal vs non-metal from k PCs (logistic regression, grouped 5-fold) ===")
chem_all = np.array(["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in d.formula])
for k in [2, 10, p]:
    steps = [StandardScaler()] + ([PCA(k)] if k < p else []) + [LogisticRegression(max_iter=5000)]
    pr_ = cross_val_predict(make_pipeline(*steps), X, metal, cv=cv, groups=chem_all)
    print(f"   k = {k:3d}: accuracy = {np.mean(pr_ == metal):.3f}")
np.savez(os.path.join(HERE, "64_pca_results.npz"), evr=evr, evr_raw=pr.explained_variance_ratio_,
         T=ps.transform(Z)[:, :3], y=y, load=ps.components_[:2], names=names)
print(f"time {time.time()-t0:.1f} s")
