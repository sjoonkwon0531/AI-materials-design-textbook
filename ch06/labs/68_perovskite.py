"""Lab 6.8: Goldschmidt t vs Bartel tau on the 576 ABX3 solids of Bartel et al. (2019),
Table S1 (authors' GitHub repository CJBartel/perovskite-stability). Recompute t and tau
from the tabulated Shannon radii and oxidation states, reproduce the accuracies, Platt-scale
tau, and compare with ML baselines (logistic, decision tree, random forest) and with a
composition-only Magpie random forest. Also leave-one-anion-out extrapolation."""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore"); t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "bartel_TableS1.csv"))
d = d.rename(columns={"rA (Ang)": "rA", "rB (Ang)": "rB", "rX (Ang)": "rX"})
y = (d.exp_label > 0).astype(int).values
tr, te = (d.is_train == 1).values, (d.is_train == -1).values
print("compounds %d: perovskite %d, non-perovskite %d; train %d / test %d" % (len(d), y.sum(), (1 - y).sum(), tr.sum(), te.sum()))
rA, rB, rX, nA = d.rA.values, d.rB.values, d.rX.values, d.nA.values
t = (rA + rX) / (np.sqrt(2) * (rB + rX))
tau = rX / rB - nA * (nA - (rA / rB) / np.log(rA / rB))
mu = rB / rX
dtau = np.abs(tau - d.tau)
print("max |t - table| = %.3f; |tau - table|: median %.3f, 99th pct %.3f, max %.2f (at rA/rB=%.3f, %s)" % (
    np.abs(t - d.t).max(), np.median(dtau), np.quantile(dtau, 0.99), dtau.max(), (rA / rB)[dtau.argmax()], d.ABX3[dtau.argmax()]))
print("all rA > rB: %s" % (rA > rB).all())
pt = ((t > 0.825) & (t < 1.059)).astype(int); pT = (tau < 4.18).astype(int)
print("agreement with table predictions: t %.3f, tau %.3f" % ((2 * pt - 1 == d.t_pred).mean(), (2 * pT - 1 == d.tau_pred).mean()))
acc = lambda p, m=slice(None): (p[m] == y[m]).mean()
print("accuracy            all     train   test")
print("t (0.825<t<1.059)  %.3f   %.3f   %.3f" % (acc(pt), acc(pt, tr), acc(pt, te)))
print("tau < 4.18         %.3f   %.3f   %.3f" % (acc(pT), acc(pT, tr), acc(pT, te)))
print("by anion   n   perov%   acc_t   acc_tau")
for X in ("O", "F", "Cl", "Br", "I"):
    m = (d.X == X).values
    print("  %-3s   %4d   %5.1f   %5.3f   %5.3f" % (X, m.sum(), 100 * y[m].mean(), acc(pt, m), acc(pT, m)))
# re-fit the thresholds on the training set only (1-D decision stumps / depth-2 tree)
dt_t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(t[tr, None], y[tr])
dt_T = DecisionTreeClassifier(max_depth=1, random_state=0).fit(tau[tr, None], y[tr])
print("refit on train: tau threshold %.3f -> test acc %.3f ; t depth-2 tree test acc %.3f, thresholds %s" % (
    dt_T.tree_.threshold[0], dt_T.score(tau[te, None], y[te]), dt_t.score(t[te, None], y[te]),
    np.round(np.sort(dt_t.tree_.threshold[dt_t.tree_.threshold > -2]), 3).tolist()))
tm = np.c_[t, mu]
dt_tm = DecisionTreeClassifier(max_depth=3, random_state=0).fit(tm[tr], y[tr])
print("(t, mu) depth-3 tree: train %.3f test %.3f" % (dt_tm.score(tm[tr], y[tr]), dt_tm.score(tm[te], y[te])))
# Platt scaling: logistic regression of the label on tau (train only)
pl = LogisticRegression(C=1e6).fit(tau[tr, None], y[tr])
a, b = pl.coef_[0, 0], pl.intercept_[0]
P = pl.predict_proba(tau[:, None])[:, 1]
print("Platt: P(perovskite) = 1/(1+exp(-(%.3f*tau %+.3f)))  -> P=0.5 at tau=%.3f" % (a, b, -b / a))
print("test: Brier %.4f  log-loss %.4f ; corr with table tau_prob %.3f, mean|diff| %.3f" % (
    brier_score_loss(y[te], P[te]), log_loss(y[te], P[te]), np.corrcoef(P, d.tau_prob)[0, 1], np.abs(P - d.tau_prob).mean()))
for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
    m = (P >= lo) & (P < hi)
    print("  P in [%.1f,%.1f): n=%3d  observed perovskite fraction %.2f" % (lo, hi, m.sum(), y[m].mean()))
# ML baselines on the six raw inputs, and with engineered (t, mu, tau)
raw = d[["rA", "rB", "rX", "nA", "nB", "nX"]].values
eng = np.c_[raw, t, mu, tau]
models = {"logistic (raw 6)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), raw),
          "tree depth 4 (raw 6)": (DecisionTreeClassifier(max_depth=4, random_state=0), raw),
          "random forest (raw 6)": (RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=1), raw),
          "random forest (raw 6 + t, mu, tau)": (RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=1), eng),
          "logistic (tau only)": (LogisticRegression(C=1e6), tau[:, None])}
# composition-only Magpie features of the ABX3 formula (no radii, no oxidation states given)
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
ep = ElementProperty.from_preset("magpie")
M = np.array([ep.featurize(Composition(f)) for f in d.ABX3])
M = np.nan_to_num(M)
models["random forest (Magpie, 132)"] = (RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=1), M)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
print("model                                  test(split)   5x5-CV mean +/- std")
res = {}
for k, (mdl, F) in models.items():
    s_te = mdl.fit(F[tr], y[tr]).score(F[te], y[te])
    s_cv = cross_val_score(mdl, F, y, cv=cv, n_jobs=1)
    res[k] = (s_te, s_cv.mean(), s_cv.std())
    print("%-38s %.3f        %.3f +/- %.3f" % (k, s_te, s_cv.mean(), s_cv.std()))
print("tau<4.18 (fixed rule) 5x5-CV-equivalent: %.3f (no fitting; single test-split value %.3f)" % (acc(pT), acc(pT, te)))
# learning curves (accuracy on the official test set; 20 random subsamples of the training set)
rng = np.random.default_rng(0); itr = np.where(tr)[0]
sizes = (20, 40, 80, 160, 320, 460); lc = {k: [] for k in ("tau stump", "RF raw 6", "RF Magpie")}
for n in sizes:
    acc_n = {k: [] for k in lc}
    for rep in range(20 if n < 460 else 1):
        s = rng.choice(itr, n, replace=False)
        if len(set(y[s])) < 2: continue
        acc_n["tau stump"].append(DecisionTreeClassifier(max_depth=1).fit(tau[s, None], y[s]).score(tau[te, None], y[te]))
        acc_n["RF raw 6"].append(RandomForestClassifier(200, min_samples_leaf=2, random_state=rep, n_jobs=1).fit(raw[s], y[s]).score(raw[te], y[te]))
        acc_n["RF Magpie"].append(RandomForestClassifier(200, min_samples_leaf=2, random_state=rep, n_jobs=1).fit(M[s], y[s]).score(M[te], y[te]))
    for k in lc: lc[k].append((np.mean(acc_n[k]), np.std(acc_n[k])))
    print("  n=%3d  tau-stump %.3f+/-%.3f   RF raw6 %.3f+/-%.3f   RF Magpie %.3f+/-%.3f" % ((n,) + sum(lc[k][-1] for k in lc), ) if False else
          "  n=%3d  tau-stump %.3f+/-%.3f   RF raw6 %.3f+/-%.3f   RF Magpie %.3f+/-%.3f" % ((n,) + tuple(v for k in lc for v in lc[k][-1])))
# leave-one-anion-family-out: train on oxides, test on halides (and the reverse)
hal = d.X.isin(["F", "Cl", "Br", "I"]).values; oxi = ~hal
print("extrapolation across anion families (accuracy):")
print("  direction            tau<4.18   tau-stump(refit)   RF raw6   RF Magpie")
for name, a_, b_ in (("oxides -> halides", oxi, hal), ("halides -> oxides", hal, oxi)):
    st = DecisionTreeClassifier(max_depth=1).fit(tau[a_, None], y[a_]).score(tau[b_, None], y[b_])
    r6 = RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=1).fit(raw[a_], y[a_]).score(raw[b_], y[b_])
    rm = RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=1).fit(M[a_], y[a_]).score(M[b_], y[b_])
    print("  %-20s %.3f      %.3f              %.3f     %.3f   (n_test=%d)" % (name, acc(pT, b_), st, r6, rm, b_.sum()))
# decomposition enthalpies (Table S2) and double perovskites (Table S3)
s2 = pd.read_csv(os.path.join(HERE, "bartel_TableS2.csv"))
stable = s2["dHdec (meV/atom)"] > 0
print("Table S2: %d compounds with PBE dH_dec; tau agrees with sign(dH_dec) for %d (%.1f%%); t for %d" % (
    len(s2), ((s2.tau_pred > 0) == stable).sum(), 100 * ((s2.tau_pred > 0) == stable).mean(), ((s2.t_pred > 0) == stable).sum()))
s3 = pd.read_csv(os.path.join(HERE, "bartel_TableS3.csv"))
print("Table S3: %d double perovskites (%s)" % (len(s3), s3.groupby(["A", "X"]).size().to_dict()))
for (A, X), g in s3.groupby(["A", "X"]):
    print("  %s2BB'%s6: n=%d, tau-predicted perovskite %d (%.1f%%), t-predicted %d (%.1f%%), P(tau)>0.8: %d" % (
        A, X, len(g), (g.tau_pred > 0).sum(), 100 * (g.tau_pred > 0).mean(), (g.t_pred > 0).sum(), 100 * (g.t_pred > 0).mean(), (g.tau_prob > 0.8).sum()))
r3 = s3.rename(columns={"rA (Ang)": "rA", "rB (Ang)": "rB", "rX (Ang)": "rX"})
r3["nB"] = (r3.nB1 + r3.nB2) / 2                     # averaged B site, as in the paper's A2BB'X6 treatment
print("Table S3: max P(tau) %.2f, median %.2f; tau range %.2f-%.2f" % (s3.tau_prob.max(), s3.tau_prob.median(), s3.tau.min(), s3.tau.max()))
rf6 = RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=1).fit(raw, y)
p_rf = rf6.predict(r3[["rA", "rB", "rX", "nA", "nB", "nX"]].values.astype(float))
print("RF(raw 6, all 576) on Table S3 averaged-B inputs: predicts perovskite %.1f%%; agrees with tau %.1f%%" % (
    100 * p_rf.mean(), 100 * (p_rf == (r3.tau_pred > 0)).mean()))
print("range check, Table S3 vs training: rA %.2f-%.2f (train %.2f-%.2f), rX %.2f-%.2f (train %.2f-%.2f)" % (
    r3.rA.min(), r3.rA.max(), rA.min(), rA.max(), r3.rX.min(), r3.rX.max(), rX.min(), rX.max()))
np.savez(os.path.join(HERE, "68_out.npz"), t=t, tau=tau, mu=mu, y=y, P=P, tr=tr, te=te, X=d.X.values.astype(str),
         sizes=np.array(sizes), lc_tau=np.array(lc["tau stump"]), lc_raw=np.array(lc["RF raw 6"]), lc_mag=np.array(lc["RF Magpie"]),
         s3_tau=s3.tau.values, s3_prob=s3.tau_prob.values, s3_A=s3.A.values.astype(str))
print("wall %.0f s" % (time.time() - t0))
