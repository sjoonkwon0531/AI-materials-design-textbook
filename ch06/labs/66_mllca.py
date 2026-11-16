"""Lab 66_mllca: ML prediction of a SYNTHETIC cradle-to-gate GWP from molecular-level descriptors,
and the extrapolation risk when the training set covers only simple (bulk) chemicals.
The 'true' law and all data are SYNTHETIC; no real LCA database is used.
ln GWP = -1.0 + 0.30*steps + 0.45*ln(MW/100) + 0.8*halogen + 0.10*n_arom + eps, eps ~ N(0, 0.35^2)."""
import os, json, time, warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)

def make(n, steps_rng, mw_rng):
    steps = rng.integers(steps_rng[0], steps_rng[1] + 1, n)
    mw = np.exp(rng.uniform(np.log(mw_rng[0]), np.log(mw_rng[1]), n))
    hal = (rng.random(n) < 0.25).astype(float)
    arom = rng.integers(0, 4, n)
    het = rng.uniform(0, 0.4, n)                       # irrelevant descriptor (heteroatom fraction)
    lnG = -1.0 + 0.30 * steps + 0.45 * np.log(mw / 100) + 0.8 * hal + 0.10 * arom + rng.normal(0, 0.35, n)
    X = np.column_stack([steps, np.log(mw / 100), hal, arom, het])
    return X, lnG

Xtr, ytr = make(400, (1, 5), (30, 300))          # 'bulk' chemicals: training domain
Xin, yin = make(400, (1, 5), (30, 300))          # in-domain test
Xout, yout = make(400, (6, 10), (300, 800))      # 'fine' chemicals: out of domain
models = {
    "linear (ln GWP)": LinearRegression(),
    "RF":              RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=2, random_state=0),
    "LightGBM":        lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, n_jobs=1,
                                         random_state=0, verbose=-1),
}
print("=== SYNTHETIC data: train n=400 (steps 1-5, MW 30-300); test in-domain n=400; OOD n=400 (steps 6-10, MW 300-800) ===")
print(f"{'model':16s} {'MAE ln in':>10s} {'MAE ln OOD':>11s} {'median ratio OOD':>17s}")
pred = {}
for k, m in models.items():
    m.fit(Xtr, ytr); pi, po = m.predict(Xin), m.predict(Xout); pred[k] = po
    print(f"{k:16s} {np.mean(np.abs(pi-yin)):10.3f} {np.mean(np.abs(po-yout)):11.3f} {np.exp(np.median(po-yout)):17.3f}")
print(f"noise floor: E|eps| = 0.35*sqrt(2/pi) = {0.35*np.sqrt(2/np.pi):.3f}")
print(f"true ln GWP: train max {ytr.max():.2f}, OOD mean {yout.mean():.2f}; RF OOD prediction max {pred['RF'].max():.2f}")

# RF spread across trees as an 'uncertainty', and a kNN distance applicability-domain score
rf = models["RF"]
def spread(Xq): return np.std([t.predict(Xq) for t in rf.estimators_], axis=0)
sc = StandardScaler().fit(Xtr); nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(Xtr))
def dist(Xq): return nn.kneighbors(sc.transform(Xq))[0].mean(axis=1)
dtr = nn.kneighbors(sc.transform(Xtr), n_neighbors=6)[0][:, 1:].mean(axis=1)   # leave self out
thr = np.percentile(dtr, 95)
s_in, s_out = spread(Xin), spread(Xout)
d_in, d_out = dist(Xin), dist(Xout)
e_in, e_out = np.abs(rf.predict(Xin) - yin), np.abs(pred["RF"] - yout)
print("=== RF tree spread vs actual error ===")
print(f"  in-domain : mean tree sd {s_in.mean():.3f}   mean |error| {e_in.mean():.3f}")
print(f"  OOD       : mean tree sd {s_out.mean():.3f}   mean |error| {e_out.mean():.3f}")
print(f"=== kNN (k=5) distance in standardised descriptor space, 95th pct of training = {thr:.3f} ===")
print(f"  flagged as outside domain: in-domain {np.mean(d_in>thr):.3f}, OOD {np.mean(d_out>thr):.3f}")
# a mixed test set: does the AD flag separate large errors?
d_all = np.r_[d_in, d_out]; e_all = np.r_[e_in, e_out]
print(f"  mixed test set: mean |error| flagged {e_all[d_all>thr].mean():.3f}  vs unflagged {e_all[d_all<=thr].mean():.3f}")
json.dump(dict(Xin=Xin[:, :2].tolist(), Xout=Xout[:, :2].tolist(), Xtr=Xtr[:, :2].tolist(),
               yin=yin.tolist(), yout=yout.tolist(), ytr=ytr.tolist(),
               pin={k: models[k].predict(Xin).tolist() for k in models}, pout={k: pred[k].tolist() for k in models},
               s_in=s_in.tolist(), s_out=s_out.tolist(), d_in=d_in.tolist(), d_out=d_out.tolist(),
               thr=float(thr), e_in=e_in.tolist(), e_out=e_out.tolist()),
          open(os.path.join(HERE, "66_mllca.json"), "w"))
