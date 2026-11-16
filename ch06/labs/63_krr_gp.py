"""Lab 6.3.3 — kernel ridge regression (RBF) vs linear ridge on the same Magpie
features (non-metal subset of matbench_expt_gap, n = 2,151, p = 123), the
primal/dual identity for a linear kernel, and a Gaussian-process preview
(predictive variance and its calibration). Deterministic; ~2-3 min on 2 cores."""
import os, time, warnings
import numpy as np, pandas as pd
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
d = pd.read_parquet(os.path.join(HERE, "magpie_expt_gap.parquet"))
d = d[d.target > 0].reset_index(drop=True)
feats = [c for c in d.columns if c.startswith("MagpieData")]
X = d[feats].to_numpy(float); y = d.target.to_numpy(float)
keep = X.std(0) > 0
frac_mode = np.array([pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(X.shape[1])])
X = X[:, keep & (frac_mode <= 0.99)]
n, p = X.shape
chemsys = np.array(["-".join(sorted(e.symbol for e in Composition(f).elements)) for f in d.formula])
print(f"n = {n}, p = {p}")

# ---------- Part A: dual = primal for the linear kernel ----------
Z = StandardScaler().fit_transform(X); yc = y - y.mean(); lam = 10.0
w = Ridge(alpha=lam, fit_intercept=False).fit(Z, yc).coef_
K = Z @ Z.T; a = np.linalg.solve(K + lam * np.eye(n), yc)
print("\n=== Part A: representer theorem check (linear kernel) ===")
print(f"primal: solve a {p}x{p} system;  dual: solve a {n}x{n} system")
print(f"max |Z w - K a| = {np.abs(Z @ w - K @ a).max():.2e};  max |w - Z^T a| = {np.abs(w - Z.T @ a).max():.2e}")

# ---------- Part B: grouped CV, ridge vs RBF-KRR ----------
gkf = list(GroupKFold(5).split(X, y, chemsys))
clus = KMeans(n_clusters=8, n_init=10, random_state=0).fit(StandardScaler().fit_transform(X)).labels_
lco = list(LeaveOneGroupOut().split(X, y, clus))
def cv_pred(make, folds):
    pr = np.empty(n)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr]); m = make().fit(sc.transform(X[tr]), y[tr] - y[tr].mean())
        pr[te] = m.predict(sc.transform(X[te])) + y[tr].mean()
    return pr
mae = lambda pr: np.mean(np.abs(pr - y))
print("\n=== Part B: grouped 5-fold CV (chemical system) ===")
grid = {}
for g in [0.001, 0.003, 0.01, 0.03]:
    for al in [0.03, 0.1, 0.3]:
        grid[(g, al)] = mae(cv_pred(lambda: KernelRidge(kernel="rbf", gamma=g, alpha=al), gkf))
print("KRR-RBF grid (gamma: MAE at alpha = 0.03 / 0.1 / 0.3):")
for g in [0.001, 0.003, 0.01, 0.03]:
    print(f"   gamma = {g:<6}: " + " / ".join(f"{grid[(g, al)]:.3f}" for al in [0.03, 0.1, 0.3]))
gb, ab = min(grid, key=grid.get)
pr_r = cv_pred(lambda: Ridge(alpha=31.6), gkf)
pr_k = cv_pred(lambda: KernelRidge(kernel="rbf", gamma=gb, alpha=ab), gkf)
sub = np.random.default_rng(0).choice(n, 400, replace=False)
D2 = ((Z[sub, None, :] - Z[None, sub, :])**2).sum(-1); med = np.median(D2[np.triu_indices(400, 1)])
print(f"median squared distance between standardized points = {med:.1f}; 1/median = {1/med:.4f}")
print(f"ridge (alpha=31.6)          : MAE = {mae(pr_r):.3f} eV, RMSE = {np.sqrt(np.mean((pr_r-y)**2)):.3f}")
print(f"KRR-RBF (gamma={gb}, alpha={ab}): MAE = {mae(pr_k):.3f} eV, RMSE = {np.sqrt(np.mean((pr_k-y)**2)):.3f}")
print("(hyperparameters chosen on the same grouped folds: the KRR number is mildly optimistic)")
pl_r = cv_pred(lambda: Ridge(alpha=31.6), lco)
pl_k = cv_pred(lambda: KernelRidge(kernel="rbf", gamma=gb, alpha=ab), lco)
print(f"leave-cluster-out (k=8): ridge MAE = {mae(pl_r):.3f}, KRR MAE = {mae(pl_k):.3f}")
# where does KRR revert to the mean? distance to training set vs kernel 'support'
kmax = np.empty(n)
for tr, te in lco:
    sc = StandardScaler().fit(X[tr]); A_, B_ = sc.transform(X[tr]), sc.transform(X[te])
    dd = ((B_[:, None, :] - A_[None, :, :])**2).sum(-1).min(1); kmax[te] = np.exp(-gb * dd)
print(f"LCO: median over test points of max_j k(x, x_j) = {np.median(kmax):.3f} "
      f"(random-like grouped CV would be closer to 1)")
far = kmax < 0.2
print(f"LCO test points with max kernel < 0.2: {far.sum()}; their KRR MAE = {np.mean(np.abs(pl_k-y)[far]):.3f}, "
      f"ridge MAE = {np.mean(np.abs(pl_r-y)[far]):.3f}, |KRR pred - train mean| median = "
      f"{np.median(np.abs(pl_k[far] - y.mean())):.3f}")

# ---------- Part C: GP preview ----------
print("\n=== Part C: Gaussian process = Bayesian KRR with a predictive variance ===")
tr, te = gkf[0]
rng = np.random.default_rng(1); tr = rng.choice(tr, 1000, replace=False)   # subsample for speed
sc = StandardScaler().fit(X[tr]); Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te]); m0 = y[tr].mean()
kern = Ck(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.1)
gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=0, random_state=0)
gp.fit(Ztr, y[tr] - m0)
mu, sd = gp.predict(Zte, return_std=True); mu += m0
print(f"train n = {len(tr)} (subsample of fold-0 training set), test n = {len(te)}")
print(f"learned kernel: {gp.kernel_}")
amp = gp.kernel_.k1.k1.constant_value; ell = gp.kernel_.k1.k2.length_scale; s2 = gp.kernel_.k2.noise_level
kr = KernelRidge(kernel="rbf", gamma=1/(2*ell**2), alpha=s2/amp).fit(Ztr, y[tr] - m0)
print(f"max |GP mean - KRR(gamma=1/2l^2, alpha=s_n^2/s_f^2)| = {np.abs(kr.predict(Zte) + m0 - mu).max():.2e}")
err = np.abs(mu - y[te])
print(f"GP test MAE = {err.mean():.3f} eV; mean predictive std = {sd.mean():.3f} eV")
for z, nom in [(1.0, 0.683), (1.96, 0.95)]:
    print(f"   coverage of mu +/- {z}*std: {np.mean(err <= z*sd):.3f} (nominal {nom})")
rho = spearmanr(sd, err).correlation
print(f"Spearman rank corr(std, |error|) = {rho:.3f}")
q = np.quantile(sd, [0, 0.25, 0.5, 0.75, 1.0])
for lo, hi in zip(q[:-1], q[1:]):
    msk = (sd >= lo) & (sd <= hi)
    print(f"   std in [{lo:.2f}, {hi:.2f}]: n = {msk.sum():3d}, MAE = {err[msk].mean():.3f}")
np.savez(os.path.join(HERE, "63_krr_gp_results.npz"), y=y, pr_r=pr_r, pr_k=pr_k, pl_r=pl_r, pl_k=pl_k,
         kmax=kmax, mu=mu, sd=sd, yte=y[te])
print(f"time {time.time()-t0:.1f} s")
