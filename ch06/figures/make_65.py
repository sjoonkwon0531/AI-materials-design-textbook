"""Figures for Sec. 6.5 (ensembles). Reads labs/65_*.json."""
import sys, json, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.join(HERE, "..", "labs")
J = lambda f: json.load(open(os.path.join(LAB, f)))

# ---------- Fig 6.5.1: gradient boosting on a 1D toy ----------
t = J("65_boost_toy.json")
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
ax = axs[0]
ax.scatter(t["x"], t["y"], s=12, color=GRAY, alpha=0.5, label="data")
ax.plot(t["xg"], t["fg"], color="k", lw=1.5, ls=":", label="true f(x)")
for (m, c, ls) in (("1", GOLD, "-"), ("10", BLUE, "--"), ("50", RED, "-"), ("500", PURPLE, "-.")):
    ax.plot(t["xg"], t["snap"][m], color=c, ls=ls, lw=2.2, label=f"M = {m}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(frameon=False, ncol=2, fontsize=14, loc="lower left")
panel_label(ax, "a")
ax = axs[1]; M = np.arange(1, len(t["tr"]) + 1)
ax.plot(M, t["tr"], color=NAVY, lw=2.2, label="train MSE")
ax.plot(M, t["te"], color=RED, lw=2.2, ls="--", label="test MSE")
ax.axhline(0.0625, color=GRAY, ls=":", lw=1.5, label="noise variance")
ax.set_xscale("log"); ax.set_xlabel("Boosting stages M"); ax.set_ylabel("MSE"); ax.legend(frameon=False)
panel_label(ax, "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.5.3_boosting_toy.png")); plt.close(fig)

# ---------- Fig 6.5.2: variance vs B and bias/variance bars ----------
b = J("65_bv.json")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axs[0]
for (name, c, mk, ls) in (("bagging (m=p)", NAVY, "o", "-"), ("RF m=p/3", RED, "s", "--"), ("RF m=p/10", GREEN, "^", "-.")):
    cv = b["curves"][name]
    ax.plot(cv["B"], cv["emp"], mk, color=c, ms=9, label=f"{name}, $\\rho$={cv['rho']:.3f}")
    Bf = np.logspace(0, 2, 100)
    ax.plot(Bf, cv["rho"] * cv["sigma2"] + (1 - cv["rho"]) * cv["sigma2"] / Bf, color=c, ls=ls, lw=1.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Number of trees B")
ax.set_ylabel("Variance of prediction (eV$^2$)"); ax.legend(frameon=False, fontsize=14)
panel_label(ax, "a")
ax = axs[1]; names = list(b["bv"]); x = np.arange(len(names))
b2 = [b["bv"][k]["bias2"] for k in names]; vv = [b["bv"][k]["var"] for k in names]
ax.bar(x, b2, color=BLUE, label="bias$^2$ + noise")
ax.bar(x, vv, bottom=b2, color=GOLD, hatch="//", edgecolor="w", label="variance")
lab = ["tree\n(full)", "tree\n(depth 4)", "bagging", "RF", "XGBoost", "LightGBM"]
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=15); ax.set_ylabel("Test MSE (eV$^2$)")
ax.legend(frameon=False, loc="upper right"); panel_label(ax, "b", x=-0.11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.5.1_bias_variance.png")); plt.close(fig)

# ---------- Fig 6.5.3: CV MAE and learning curves ----------
e = J("65_ensemble.json"); lc = J("65_lc.json")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
ax = axs[0]
names = ["dummy", "tree", "RF", "XGBoost", "LightGBM", "stack (OOF)", "stack (naive)"]
lab = ["dummy", "tree", "RF", "XGB", "LGBM", "stack\nOOF", "stack\nnaive"]
mae = [e["reg"][k]["mae"] for k in names]; sd = [e["reg"][k]["sd"] for k in names]
cols = [GRAY, GOLD, NAVY, BLUE, BLUE, RED, PURPLE]
ax.bar(range(len(names)), mae, yerr=sd, color=cols, capsize=4)
for i, v in enumerate(mae):
    ax.text(i, v + sd[i] + 0.02, f"{v:.3f}", ha="center", fontsize=13)
ax.set_xticks(range(len(names))); ax.set_xticklabels(lab, fontsize=14); ax.set_ylabel("5-fold CV MAE (eV)")
panel_label(ax, "a")
ax = axs[1]; n = lc["sizes"]
for k, c, mk, ls in (("tree", GOLD, "^", "-."), ("RF", NAVY, "o", "-"), ("LightGBM", RED, "s", "--")):
    te = np.array(lc["te"][k]); tr = np.array(lc["tr"][k])
    ax.errorbar(n, te.mean(0), yerr=te.std(0, ddof=1), color=c, marker=mk, ls=ls, lw=2, ms=8, capsize=3, label=k)
    ax.plot(n, tr.mean(0), color=c, marker=mk, ls=ls, lw=1, ms=5, mfc="none", alpha=0.6)
ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(q) for q in n]); ax.minorticks_off(); ax.set_xlabel("Training-set size n"); ax.set_ylabel("MAE (eV)")
ax.legend(frameon=False, loc="upper right", fontsize=14); panel_label(ax, "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.5.4_cv_learning.png")); plt.close(fig)

# ---------- Fig 6.5.4: MDI vs permutation importance ----------
im = J("65_importance.json")
k = list(im["mdi"]); mdi = np.array([im["mdi"][q] for q in k]); per = np.array([im["perm"][q] for q in k])
fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.scatter(mdi, np.clip(per, 1e-5, None), s=30, color=NAVY, alpha=0.6)
i = k.index("PROBE noise")
ax.scatter([mdi[i]], [max(per[i], 1e-5)], s=160, marker="*", color=RED, zorder=5)
ax.annotate("pure-noise probe", (mdi[i], max(per[i], 1e-5)), xytext=(10, 8), textcoords="offset points", color=RED, fontsize=15)
for j in np.argsort(-per)[:1]:
    ax.annotate(k[j], (mdi[j], per[j]), xytext=(-150, -6), textcoords="offset points", fontsize=12.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Impurity importance (MDI)"); ax.set_ylabel("Permutation importance (eV MAE)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.5.2_importance.png")); plt.close(fig)
print("done")
