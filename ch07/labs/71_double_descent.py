"""Lab 7.1.3b -- double descent with random ReLU features on the Magpie band-gap data.
Two-layer network whose first layer is random and frozen; only the output layer is fit by
minimum-norm least squares (lstsq = pseudo-inverse, Ch.1 sec 1.5.6) or by ridge.
Run: python 71_double_descent.py   (about 1 min)"""
import time, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
T0 = time.time()
df = pd.read_parquet("../data/magpie_expt_gap.parquet")
X = df.drop(columns=["formula", "target"]).values; y = df.target.values
tr, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
rng = np.random.default_rng(0)
n = 600
sub = rng.choice(tr, n, replace=False)
vt = VarianceThreshold(1e-6).fit(X[sub]); sc = StandardScaler().fit(vt.transform(X[sub]))
f = lambda i: np.clip(sc.transform(vt.transform(X[i])), -5, 5)
Xs, Xt = f(sub), f(te); ys, yt = y[sub], y[te]
p = Xs.shape[1]; ym = ys.mean()
Ns = [30, 100, 300, 450, 550, 590, 600, 610, 650, 750, 1000, 2000, 5000, 10000, 30000]
print(f"train n={n} (random subset of fold-0 train), test {len(te)} (fold-0 test), p={p} standardized Magpie features")
print(f"{'N feat':>7s} {'train RMSE':>10s} {'test RMSE minnorm':>18s} {'||w||':>9s} {'test RMSE ridge 10':>21s}")
R = {"N": Ns, "tr": [], "te": [], "w": [], "ridge": []}
for N in Ns:
    trs, tes, ws, rid = [], [], [], []
    for s in range(3):
        W = np.random.default_rng(100 + s).normal(0, 1 / np.sqrt(p), (p, N))
        Phi, Phit = np.maximum(Xs @ W, 0), np.maximum(Xt @ W, 0)
        K, Kt = Phi @ Phi.T, Phit @ Phi.T                          # dual (kernel) form, n x n
        alpha = np.linalg.pinv(K, rcond=1e-12, hermitian=True) @ (ys - ym)   # w = Phi^T K^+ y = pinv(Phi) y (min-norm)
        trs.append(np.sqrt(np.mean((K @ alpha + ym - ys) ** 2)))
        tes.append(np.sqrt(np.mean((np.clip(Kt @ alpha + ym, 0, None) - yt) ** 2)))
        ws.append(np.sqrt(max(alpha @ K @ alpha, 0)))
        lam = 10.0                                                 # ridge, same dual form
        al2 = np.linalg.solve(K + lam * np.eye(n), ys - ym)
        rid.append(np.sqrt(np.mean((np.clip(Kt @ al2 + ym, 0, None) - yt) ** 2)))
    for k, v in zip(["tr", "te", "w", "ridge"], [trs, tes, ws, rid]): R[k].append(np.median(v))
    print(f"{N:7d} {np.median(trs):10.3f} {np.median(tes):18.3f} {np.median(ws):9.2e} {np.median(rid):21.3f}")
print("(medians over 3 random first layers; predictions clipped at 0)")
print(f"baseline (predict train mean): test RMSE {np.sqrt(np.mean((ym-yt)**2)):.3f}")
np.savez("71_double_descent.npz", **{k: np.array(v) for k, v in R.items()}, n=n)
print(f"wall {time.time()-T0:.0f} s")
