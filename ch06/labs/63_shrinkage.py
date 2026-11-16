"""Lab 6.3.2 — ridge, LASSO and elastic net on Magpie features of the non-metal
subset of matbench_expt_gap (n = 2,151). Checks the SVD form of ridge, implements
coordinate-descent LASSO from scratch and compares with scikit-learn, computes
regularization paths and grouped CV curves, and asks which features survive.
Deterministic; runtime ~1-2 min on 2 cores."""
import os, time, warnings
import numpy as np, pandas as pd
from pymatgen.core import Composition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, lasso_path, enet_path
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
d = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
d = d[d.target > 0].reset_index(drop=True)
feats = [c for c in d.columns if c.startswith("MagpieData")]
X = d[feats].to_numpy(float); y = d.target.to_numpy(float)
names = np.array([f.replace("MagpieData ", "") for f in feats])
keep = X.std(0) > 0
frac_mode = np.array([pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(X.shape[1])])
keep &= frac_mode <= 0.99                      # same filter as Lab 6.3.1
X, names = X[:, keep], names[keep]
n, p = X.shape
chemsys = np.array(["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in d.formula])
print(f"n = {n}, p = {p}")

# ---------- Part A: ridge = SVD shrinkage ----------
Z = StandardScaler().fit_transform(X); yc = y - y.mean()
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
print("\n=== Part A: ridge through the SVD ===")
for lam in [1.0, 100.0]:
    f = s**2 / (s**2 + lam)
    w_svd = Vt.T @ (f / np.where(s > 1e-10 * s[0], s, np.inf) * (U.T @ yc))
    w_skl = Ridge(alpha=lam, fit_intercept=False).fit(Z, yc).coef_
    print(f"lambda = {lam:6.1f}: max|w_SVD - w_sklearn| = {np.abs(w_svd - w_skl).max():.2e};  "
          f"df(lambda) = sum f_i = {f.sum():6.2f};  ||w|| = {np.linalg.norm(w_svd):.3f}")
print(f"sigma_1^2 = {s[0]**2:.1f}, sigma_50^2 = {s[49]**2:.2f}, sigma_100^2 = {s[99]**2:.2e}; "
      f"#(sigma_i^2 > 100) = {(s**2 > 100).sum()}")

# grouped CV for ridge over lambda
gkf = list(GroupKFold(5).split(X, y, chemsys))
def cv_mae(make, X=X):
    e = np.empty(n)
    for tr, te in gkf:
        sc = StandardScaler().fit(X[tr]); m = make().fit(sc.transform(X[tr]), y[tr])
        e[te] = np.abs(m.predict(sc.transform(X[te])) - y[te])
    fold = np.array([e[te].mean() for _, te in gkf])
    return e.mean(), fold.std(ddof=1) / np.sqrt(len(gkf))
lams = np.logspace(-3, 4, 15)
rid = np.array([cv_mae(lambda: Ridge(alpha=l)) for l in lams])
i = rid[:, 0].argmin()
print("ridge grouped 5-fold CV MAE (eV) vs lambda:")
print("  " + "  ".join(f"{l:.0e}:{m:.3f}" for l, m in zip(lams[::2], rid[::2, 0])))
print(f"  best lambda = {lams[i]:.2e}, MAE = {rid[i,0]:.3f} +/- {rid[i,1]:.3f}")

print(f"[elapsed {time.time()-t0:.0f} s]")
# ---------- Part B: coordinate descent LASSO from scratch ----------
from numba import njit
@njit(cache=True)
def soft(z, g): return np.sign(z) * max(abs(z) - g, 0.0)
@njit(cache=True)
def lasso_cd(Z, y, alpha, n_iter=5000, tol=1e-9):
    """min (1/2n)||y - Z w||^2 + alpha ||w||_1 ; columns of Z standardized."""
    n, p = Z.shape; w = np.zeros(p); r = y.copy(); cs = (Z**2).sum(0) / n
    it = 0
    for it in range(n_iter):
        wmax = 0.0
        for j in range(p):
            wj = w[j]; rho = np.dot(Z[:, j], r) / n + cs[j] * wj
            w[j] = soft(rho, alpha) / cs[j]
            if w[j] != wj:
                r -= Z[:, j] * (w[j] - wj); wmax = max(wmax, abs(w[j] - wj))
        if wmax < tol: break
    return w, it + 1
print("\n=== Part B: coordinate-descent LASSO (from scratch) vs scikit-learn ===")
alpha_max = np.abs(Z.T @ yc).max() / n
print(f"alpha_max = max_j |z_j^T y|/n = {alpha_max:.4f} (all weights zero above it)")
for a in [0.1, 0.02]:
    w_cd, it = lasso_cd(np.asfortranarray(Z), yc, a)
    w_sk = Lasso(alpha=a, fit_intercept=False, tol=1e-8, max_iter=50000).fit(Z, yc).coef_
    obj = lambda w: 0.5*np.mean((yc - Z@w)**2) + a*np.abs(w).sum()
    print(f"alpha = {a}: sweeps = {it}, nonzero = {np.sum(w_cd != 0)}, objective CD = {obj(w_cd):.10f}, sklearn = {obj(w_sk):.10f}, "
          f"max|w_CD - w_sklearn| = {np.abs(w_cd - w_sk).max():.2e}")

print(f"[elapsed {time.time()-t0:.0f} s]")
# ---------- Part C: paths and grouped CV ----------
alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max) - 3, 25)
_, coefs_l, _ = lasso_path(Z, yc, alphas=alphas, max_iter=5000)
_, coefs_e, _ = enet_path(Z, yc, alphas=alphas, l1_ratio=0.5, max_iter=5000)
ridge_coefs = np.array([Vt.T @ (s / (s**2 + l) * (U.T @ yc)) for l in lams]).T
las = np.array([cv_mae(lambda: Lasso(alpha=a, max_iter=5000)) for a in alphas])
ena = np.array([cv_mae(lambda: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000)) for a in alphas])
nnz_l = (np.abs(coefs_l) > 0).sum(0); nnz_e = (np.abs(coefs_e) > 0).sum(0)
for nm, cv, nnz in [("LASSO", las, nnz_l), ("EN(0.5)", ena, nnz_e)]:
    k = cv[:, 0].argmin()
    k1 = np.where(cv[:, 0] <= cv[k, 0] + cv[k, 1])[0].min()     # one-standard-error rule: largest alpha
    print(f"{nm:8s}: best alpha = {alphas[k]:.2e}, MAE = {cv[k,0]:.3f} +/- {cv[k,1]:.3f}, nonzero = {nnz[k]};"
          f"  1-SE alpha = {alphas[k1]:.2e}, MAE = {cv[k1,0]:.3f}, nonzero = {nnz[k1]}")
k1 = np.where(las[:, 0] <= las[las[:, 0].argmin(), 0] + las[las[:, 0].argmin(), 1])[0].min()
a1se = alphas[k1]
# order of entry along the LASSO path
entry = [np.argmax(np.abs(coefs_l[j]) > 0) if np.any(coefs_l[j] != 0) else 999 for j in range(p)]
order = np.argsort(entry)
print("first 8 features to enter the LASSO path (alpha at entry):")
for j in order[:8]:
    print(f"   {names[j]:28s} alpha = {alphas[entry[j]]:.3e}")
w1 = Lasso(alpha=a1se, max_iter=5000).fit(Z, y).coef_
top = np.argsort(-np.abs(w1))[:8]
print(f"LASSO at the 1-SE alpha ({a1se:.2e}) on all data: {np.sum(w1 != 0)} nonzero; largest |w|:")
for j in top:
    print(f"   {names[j]:28s} w = {w1[j]:+.3f}")

print(f"[elapsed {time.time()-t0:.0f} s]")
# ---------- Part D: stability of selection under bootstrap ----------
rng = np.random.default_rng(0); B = 50
sel_l = np.zeros(p); sel_e = np.zeros(p)
for b in range(B):
    ib = rng.integers(0, n, n); Zb = StandardScaler().fit_transform(X[ib])
    sel_l += Lasso(alpha=a1se, max_iter=5000).fit(Zb, y[ib]).coef_ != 0
    sel_e += ElasticNet(alpha=a1se, l1_ratio=0.5, max_iter=5000).fit(Zb, y[ib]).coef_ != 0
sel_l /= B; sel_e /= B
print(f"\n=== Part D: selection frequency over {B} bootstrap resamples (alpha = {a1se:.2e}) ===")
print(f"LASSO: features selected in >=90% of resamples: {np.sum(sel_l >= 0.9)}; "
      f"in 10-90%: {np.sum((sel_l > 0.1) & (sel_l < 0.9))}")
print(f"EN   : features selected in >=90% of resamples: {np.sum(sel_e >= 0.9)}; "
      f"in 10-90%: {np.sum((sel_e > 0.1) & (sel_e < 0.9))}")
C = np.corrcoef(Z.T)
print("LASSO features with freq >= 0.9: " + ", ".join(f"{names[j]} ({sel_l[j]:.2f})" for j in np.argsort(-sel_l) if sel_l[j] >= 0.9))
for nm in ["mean Electronegativity", "maximum Electronegativity", "mean NpValence", "mean Column"]:
    if nm in names:
        j = np.where(names == nm)[0][0]
        print(f"   {nm:28s} LASSO freq {sel_l[j]:.2f}  EN freq {sel_e[j]:.2f}")
jA = np.where(names == "mean Electronegativity")[0][0]; jB = np.where(names == "mean Column")[0][0]
print(f"corr(mean Electronegativity, mean Column) = {C[jA, jB]:.3f}")
jM = np.where(names == "maximum Electronegativity")[0][0]
cc = np.abs(C[jM]).copy(); cc[jM] = 0; jP = cc.argmax()
print(f"maximum Electronegativity: most correlated partner = {names[jP]} (corr {C[jM, jP]:+.3f}), "
      f"its LASSO freq {sel_l[jP]:.2f}")
print(f"pairs of features with |corr| > 0.95: {int((np.abs(C[np.triu_indices(p, 1)]) > 0.95).sum())}")
np.savez(os.path.join(HERE, "63_shrinkage_results.npz"), lams=lams, rid=rid, alphas=alphas,
         las=las, ena=ena, coefs_l=coefs_l, coefs_e=coefs_e, ridge_coefs=ridge_coefs,
         nnz_l=nnz_l, nnz_e=nnz_e, a1se=a1se, names=names, sel_l=sel_l, sel_e=sel_e, s=s)
print(f"time {time.time()-t0:.1f} s")
