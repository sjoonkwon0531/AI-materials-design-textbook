import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
df = pd.read_parquet("../../../ch07/data/magpie_expt_gap.parquet")
X = df.drop(columns=["formula", "target"]).values; y = df.target.values
trn, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
rng = np.random.default_rng(0); n = 600; sub = rng.choice(trn, n, replace=False)
vt = VarianceThreshold(1e-6).fit(X[sub]); sc = StandardScaler().fit(vt.transform(X[sub]))
f = lambda i: np.clip(sc.transform(vt.transform(X[i])), -5, 5)
Xs, Xt = f(sub), f(te); ym = y[sub].mean(); ys = y[sub]-ym; p = Xs.shape[1]
for N in [590, 600, 610]:
    ex, lab, nt = [], [], []
    for s in range(3):
        W = np.random.default_rng(100+s).normal(0, 1/np.sqrt(p), (p, N)); Phi = np.maximum(Xs@W, 0); Phit = np.maximum(Xt@W, 0)
        U, S, Vt = np.linalg.svd(Phi, full_matrices=False); w = Vt.T@((U.T@ys)/S)
        ex.append(np.sqrt(np.mean((np.clip(Phit@w+ym, 0, None)-y[te])**2)))
        K = Phi@Phi.T; ev = np.linalg.eigvalsh(K); nt.append(int((ev < 1e-12*ev.max()).sum()))
        al = np.linalg.pinv(K, rcond=1e-12, hermitian=True)@ys; lab.append(np.sqrt(np.mean((np.clip(Phit@Phi.T@al+ym, 0, None)-y[te])**2)))
    print(N, "test RMSE exact SVD median", np.median(ex), ex, " Lab-style", np.median(lab), " #eigs truncated", nt)
