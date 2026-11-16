"""Lab 6.7 step 3: honest end-to-end pipeline on a chemical-system-grouped hold-out.
(a) small grouped hyper-parameter search, (b) direct regression vs two-stage
classifier+regressor, (c) error by chemistry family, (d) permutation importance,
(e) probability calibration of the metal classifier, (f) interval calibration
(LightGBM quantile regression and random-forest spread)."""
import os, time, warnings, itertools
import numpy as np, pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
warnings.filterwarnings("ignore"); t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "67_features_clean.parquet"))
feat = [c for c in df.columns if c not in ("formula", "target", "chemsys", "reduced")]
X, y, g = df[feat].values, df.target.values, df.chemsys.values
FAM = {**{e: "halide" for e in ("F", "Cl", "Br", "I")}, "O": "oxide",
       **{e: "chalcogenide" for e in ("S", "Se", "Te")}, **{e: "pnictide" for e in ("N", "P", "As", "Sb")}}
def family(f):
    el = max(Composition(f).elements, key=lambda e: e.X if e.X == e.X else 0)
    return FAM.get(el.symbol, "other")
fam = np.array([family(f) for f in df.formula])
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
print("train %d / test %d (chemical systems disjoint: %s)" % (len(tr), len(te), len(set(g[tr]) & set(g[te])) == 0))
base = dict(n_estimators=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=1, verbose=-1, random_state=0)
# (a) grouped grid search on the training part only
grid = list(itertools.product((15, 63), (0.03, 0.1), (5, 20)))
best = None
for nl, lr, mcs in grid:
    sc = []
    for a, b in GroupKFold(3).split(X[tr], y[tr], g[tr]):
        m = lgb.LGBMRegressor(num_leaves=nl, learning_rate=lr, min_child_samples=mcs, **base).fit(X[tr][a], y[tr][a])
        sc.append(mean_absolute_error(y[tr][b], np.clip(m.predict(X[tr][b]), 0, None)))
    print("  num_leaves=%2d lr=%.2f min_child=%2d -> grouped-CV MAE %.3f" % (nl, lr, mcs, np.mean(sc)))
    if best is None or np.mean(sc) < best[0]: best = (np.mean(sc), dict(num_leaves=nl, learning_rate=lr, min_child_samples=mcs))
hp = best[1]; print("selected:", hp)
# (b) direct vs two-stage
direct = lgb.LGBMRegressor(**hp, **base).fit(X[tr], y[tr])
yd = np.clip(direct.predict(X[te]), 0, None)
clf = lgb.LGBMClassifier(**hp, **base).fit(X[tr], (y[tr] > 0).astype(int))
p_ins = clf.predict_proba(X[te])[:, 1]
pos = tr[y[tr] > 0]
reg = lgb.LGBMRegressor(**hp, **base).fit(X[pos], y[pos])
yr = np.clip(reg.predict(X[te]), 0, None)
y2 = np.where(p_ins > 0.5, yr, 0.0)          # hard gate
ys = p_ins * yr                               # soft gate (expected value)
met, ins = y[te] == 0, y[te] > 0
def rep(name, yp):
    print("%-26s MAE all %.3f | metals %.3f | non-metals %.3f | false-metal(gap>0 predicted 0) %d" % (
        name, mean_absolute_error(y[te], yp), mean_absolute_error(y[te][met], yp[met]),
        mean_absolute_error(y[te][ins], yp[ins]), int(((yp == 0) & ins).sum())))
rep("direct regression", yd); rep("two-stage (hard gate)", y2); rep("two-stage (soft p*gap)", ys)
print("classifier: accuracy %.3f  ROC-AUC %.3f  | regressor MAE on true non-metals %.3f" % (
    accuracy_score(ins, p_ins > 0.5), roc_auc_score(ins, p_ins), mean_absolute_error(y[te][ins], yr[ins])))
print("direct regression on metals: fraction predicted < 0.1 eV = %.2f" % (yd[met] < 0.1).mean())
# (c) by family (two-stage hard gate)
print("family          n_test  mean_gap  MAE_direct  MAE_2stage")
for f in ("oxide", "chalcogenide", "halide", "pnictide", "other"):
    k = fam[te] == f
    print("%-14s %6d %9.2f %11.3f %11.3f" % (f, k.sum(), y[te][k].mean(), mean_absolute_error(y[te][k], yd[k]),
          mean_absolute_error(y[te][k], y2[k])))
# (d) permutation importance of the non-metal regressor on the test non-metals
ti = te[ins]
pi = permutation_importance(reg, X[ti], y[ti], scoring="neg_mean_absolute_error", n_repeats=5, random_state=0, n_jobs=1)
order = np.argsort(-pi.importances_mean)[:8]
print("permutation importance (increase in MAE, eV), top 8:")
for i in order: print("   %-48s %.3f +/- %.3f" % (feat[i], pi.importances_mean[i], pi.importances_std[i]))
gi = np.argsort(-reg.feature_importances_)[:5]
print("LightGBM split-count top 5:", [feat[i] for i in gi])
# (e) classifier calibration: raw vs isotonic (grouped inner CV)
def ece(p, t, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(p[b == k].mean() - t[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())
cal = CalibratedClassifierCV(lgb.LGBMClassifier(**hp, **base), method="isotonic",
                             cv=list(GroupKFold(3).split(X[tr], y[tr], g[tr]))).fit(X[tr], (y[tr] > 0).astype(int))
p_cal = cal.predict_proba(X[te])[:, 1]
print("P(non-metal) calibration: Brier raw %.4f -> isotonic %.4f ; ECE raw %.4f -> isotonic %.4f" % (
    brier_score_loss(ins, p_ins), brier_score_loss(ins, p_cal), ece(p_ins, ins.astype(float)), ece(p_cal, ins.astype(float))))
# (f) intervals on non-metals: quantile LightGBM and RF spread
q = {}
for a in (0.05, 0.5, 0.95):
    q[a] = lgb.LGBMRegressor(objective="quantile", alpha=a, **hp, **base).fit(X[pos], y[pos]).predict(X[ti])
cov = np.mean((y[ti] >= q[0.05]) & (y[ti] <= q[0.95]))
print("quantile LGBM nominal 90%% interval: coverage %.3f, mean width %.2f eV" % (cov, np.mean(q[0.95] - q[0.05])))
# split-conformal rescaling of the RF spread: calibrate on a group-disjoint slice of the TRAINING set
pa, pc = next(GroupShuffleSplit(1, test_size=0.25, random_state=7).split(pos, groups=g[pos]))
pa, pc = pos[pa], pos[pc]
rf = RandomForestRegressor(300, max_features=0.3, min_samples_leaf=2, n_jobs=1, random_state=0).fit(X[pa], y[pa])
def spread(idx):
    T = np.stack([t.predict(X[idx]) for t in rf.estimators_]); return T.mean(0), T.std(0)
mu, sd = spread(ti); muc, sdc = spread(pc)
z = np.abs(y[ti] - mu) / sd
from scipy.stats import norm
print("RF (fit on %d, calibrate on %d group-disjoint): MAE non-metals %.3f; mean sigma %.2f eV" % (
    len(pa), len(pc), mean_absolute_error(y[ti], mu), sd.mean()))
zc = np.abs(y[pc] - muc) / sdc
levels = (0.5, 0.68, 0.9, 0.95)
obs_raw, obs_cal = [], []
for lev in levels:
    qc = np.quantile(zc, lev)                      # conformal quantile of |r|/sigma
    obs_raw.append(np.mean(z <= norm.ppf(0.5 + lev / 2))); obs_cal.append(np.mean(z <= qc))
    print("   nominal %.2f -> Gaussian(sigma) observed %.3f | conformal (k=%.2f) observed %.3f" % (
        lev, obs_raw[-1], qc, obs_cal[-1]))
print("Spearman(|error|, sigma) = %.3f" % pd.Series(np.abs(y[ti] - mu)).corr(pd.Series(sd), method="spearman"))
np.savez(os.path.join(HERE, "67_pipeline_out.npz"), y=y[te], yd=yd, y2=y2, p=p_ins, pcal=p_cal, fam=fam[te],
         yti=y[ti], mu=mu, sd=sd, q05=q[0.05], q95=q[0.95], q50=q[0.5], levels=levels, obs_raw=obs_raw, obs_cal=obs_cal)
print("wall %.0f s" % (time.time() - t0))
