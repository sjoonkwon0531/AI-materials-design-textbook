"""Lab 6.3.1 — linear regression on Magpie features, four solvers, and how the
split scheme changes the reported error (random K-fold vs grouped vs leave-cluster-out).
Data: matbench_expt_gap (4,601 compositions), Magpie ElementProperty cache
(labs/magpie_expt_gap.parquet, 132 features). Regression is done on the
non-metals (gap > 0) only. Seeds fixed; runtime ~1 min on 2 cores."""
import os, time, warnings
import numpy as np, pandas as pd
from pymatgen.core import Composition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, cross_val_predict

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

def load(nonmetal_only=True):
    d = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
    if nonmetal_only:
        d = d[d.target > 0].reset_index(drop=True)
    feats = [c for c in d.columns if c.startswith("MagpieData")]
    X = d[feats].to_numpy(float); y = d.target.to_numpy(float)
    keep = X.std(0) > 0                      # drop constant columns
    return d, X[:, keep], y, [f.replace("MagpieData ", "") for f, k in zip(feats, keep) if k]

d, X, y, names = load()
print(f"non-metal subset: n = {len(y)}, p = {X.shape[1]} (after dropping constant columns)")
print(f"gap: mean {y.mean():.3f} eV, std {y.std():.3f} eV, min {y.min():.2f}, max {y.max():.2f}")

# ---------- Part A: four solvers, conditioning ----------
Xs = StandardScaler().fit_transform(X)
A = np.column_stack([Xs, np.ones(len(y))])
s = np.linalg.svd(A, compute_uv=False)
print("\n=== Part A: OLS on standardized Magpie features ===")
print(f"cond(A) = {s[0]/s[-1]:.3e}   cond(A^T A) = {(s[0]/s[-1])**2:.3e}")
print(f"singular values: largest {s[0]:.1f}, 10 smallest: " + ", ".join(f"{v:.2e}" for v in s[-10:]))
print(f"number of sigma_i < 1e-6 * sigma_1: {(s < 1e-6*s[0]).sum()}")
rank = np.linalg.matrix_rank(A)
print(f"numerical rank of A = {rank} of {A.shape[1]} columns")
try:
    w_ne = np.linalg.solve(A.T @ A, A.T @ y); print("normal eq.: solved")
except np.linalg.LinAlgError as e:
    w_ne = None; print(f"normal eq.: np.linalg.solve raised LinAlgError ('{e}')")
Q, R = np.linalg.qr(A); w_qr = np.linalg.solve(R, Q.T @ y)
print(f"QR: min |R_ii| = {np.abs(np.diag(R)).min():.2e}")
w_svd, *_ = np.linalg.lstsq(A, y, rcond=None)
for nm, w in [("QR (no pivoting)", w_qr), ("SVD/lstsq (min-norm)", w_svd)]:
    r = A @ w - y
    print(f"{nm:21s}: train RMSE = {np.sqrt(np.mean(r**2)):.4f} eV   ||w|| = {np.linalg.norm(w):.3e}"
          f"   max|w| = {np.abs(w).max():.3e}")
# null-space direction: adding it changes w but not the fit
Vt = np.linalg.svd(A, full_matrices=False)[2]; v0 = Vt[-1]
w2 = w_svd + 10.0 * v0
print(f"w_SVD + 10*v_null: ||w|| = {np.linalg.norm(w2):.3f}, max |A w2 - A w_SVD| = {np.abs(A@w2 - A@w_svd).max():.2e}")
# the redundancy: range = max - min, exactly
i_rng = names.index("range Number"); i_max = names.index("maximum Number"); i_min = names.index("minimum Number")
print("check exact collinearity: max |range - (max - min)| for Number =",
      f"{np.abs(X[:, i_rng] - (X[:, i_max] - X[:, i_min])).max():.1e}")

# ---------- Part B: split schemes ----------
chemsys = np.array(["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in d.formula])
ug, cnt = np.unique(chemsys, return_counts=True)
print("\n=== Part B: how the split changes the error ===")
print(f"chemical systems: {len(ug)} distinct; {np.sum(cnt > 1)} systems hold >1 entry, "
      f"covering {cnt[cnt > 1].sum()} of {len(y)} rows; largest: {ug[cnt.argmax()]} ({cnt.max()})")

# quasi-constant columns: most frequent value shared by > 99% of rows
frac_mode = np.array([pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(X.shape[1])])
qc = frac_mode > 0.99
print(f"quasi-constant features (>99% identical): {qc.sum()} -> " + ", ".join(np.array(names)[qc]))
Xf = X[:, ~qc]
models = {
    "ridge, 130 feat.": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X),
    "ridge, 123 feat.": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), Xf),
    "1-NN, 123 feat.": (make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1)), Xf),
    "5-NN, 123 feat.": (make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")), Xf),
}
km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(StandardScaler().fit_transform(Xf))
clus = km.labels_
print("leave-cluster-out: KMeans(k=8) on standardized features; cluster sizes:",
      np.bincount(clus).tolist())
schemes = {
    "random 5-fold": (KFold(5, shuffle=True, random_state=0), None),
    "group 5-fold (chem. system)": (GroupKFold(5), chemsys),
    "leave-cluster-out (k=8)": (LeaveOneGroupOut(), clus),
}
res = {}
print(f"{'model':18s}" + "".join(f"{k:>30s}" for k in schemes))
for mn, (m, XX) in models.items():
    row = []
    for sn, (cv, g) in schemes.items():
        p = cross_val_predict(m, XX, y, cv=cv, groups=g, n_jobs=2)
        row.append(np.mean(np.abs(p - y)))
        if mn.startswith("ridge, 130") and sn.startswith("random"):
            e = np.abs(p - y); k = e.argmax()
            worst = (d.formula[k], e[k], np.median(e), np.mean(np.delete(e, k)))
    res[mn] = row
    print(f"{mn:18s}" + "".join(f"{v:30.3f}" for v in row))
print(f"ridge/130, random 5-fold: worst point {worst[0]} |error| = {worst[1]:.3e} eV; "
      f"median AE = {worst[2]:.3f}; MAE without it = {worst[3]:.3f}")
print(f"(baseline: predict the training mean -> MAE = {np.mean(np.abs(y - y.mean())):.3f} eV)")

# nearest-neighbour distance: how close is the closest training point in each scheme?
from sklearn.neighbors import NearestNeighbors
def nn_dist(cv, g):
    out = np.empty(len(y))
    for tr, te in cv.split(Xf, y, g):
        Zf = StandardScaler().fit(Xf[tr]); nn = NearestNeighbors(n_neighbors=1).fit(Zf.transform(Xf[tr])); out[te] = nn.kneighbors(Zf.transform(Xf[te]))[0][:, 0]
    return out
print("median distance (standardized feature space) from a test point to its nearest training point:")
nnd = {}
for sn, (cv, g) in schemes.items():
    nnd[sn] = nn_dist(cv, g); print(f"   {sn:30s}: {np.median(nnd[sn]):.3f}")
np.savez(os.path.join(HERE, "63_linreg_cv_results.npz"), schemes=list(schemes), models=list(models),
         mae=np.array([res[m] for m in models]), s=s,
         **{f"nnd{i}": nnd[k] for i, k in enumerate(schemes)})
print(f"time {time.time()-t0:.1f} s")
