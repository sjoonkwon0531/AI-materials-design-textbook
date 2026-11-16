"""Figures for Section 6.3. Reads labs/63_*_results.npz written by the lab scripts."""
import sys, os, logging
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.join(HERE, "..", "labs")

# ---------- Fig 6.3.1: split schemes ----------
R = np.load(os.path.join(LAB, "63_linreg_cv_results.npz"), allow_pickle=True)
mae = R["mae"]; models = list(R["models"])
fig, axs = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
labels = ["random\n5-fold", "group 5-fold\n(chem. system)", "leave-cluster-\nout (k = 8)"]
sel = [1, 2, 3]; cols = [NAVY, RED, BLUE]; hat = [None, "//", ".."]
w = 0.25; xx = np.arange(3)
for k, (i, c, h) in enumerate(zip(sel, cols, hat)):
    ax.bar(xx + (k - 1) * w, mae[i], w, color=c, hatch=h, edgecolor="white",
           label=models[i].replace(", 123 feat.", ""))
ax.axhline(1.074, color=GRAY, ls="--", lw=1.4)
ax.text(2.45, 1.03, "predict mean", ha="right", va="top", fontsize=14, color=GRAY)
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=14); ax.set_ylabel("CV MAE (eV)")
ax.set_ylim(0, 1.45); ax.legend(loc="upper left", frameon=False, fontsize=14, ncol=3)
panel_label(ax, "a")
ax = axs[1]
names = ["random 5-fold", "group 5-fold", "leave-cluster-out"]
bins = np.linspace(0, 16, 50)
for i, (nm, c, ls) in enumerate(zip(names, [NAVY, RED, BLUE], ["-", "--", ":"])):
    v = R[f"nnd{i}"]
    ax.hist(v, bins=bins, histtype="step", lw=2.4, color=c, ls=ls, density=True,
            label=f"{nm} (median {np.median(v):.1f})")
ax.set_xlabel("Distance to nearest training point"); ax.set_ylabel("Density")
ax.legend(frameon=False, fontsize=13.5, loc="upper right")
panel_label(ax, "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.3.1_split_leakage.png")); plt.close(fig)

# ---------- Fig 6.3.2: regularization paths ----------
S = np.load(os.path.join(LAB, "63_shrinkage_results.npz"), allow_pickle=True)
names = S["names"]; alphas = S["alphas"]; lams = S["lams"]
fig = plt.figure(figsize=(15, 10.5))
sf = fig.subfigures(2, 1, hspace=0.04)
top = sf[0].subplots(1, 2); sf[0].subplots_adjust(left=0.08, right=0.98, wspace=0.28, bottom=0.14, top=0.94)
bot = sf[1].subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 0.78])); sf[1].subplots_adjust(left=0.08, right=0.98, wspace=0.75, bottom=0.14, top=0.90)
axs = np.array([top, bot])
ax = axs[0, 0]
rc = S["ridge_coefs"]
for j in range(rc.shape[0]):
    ax.plot(lams, rc[j], color=GRAY, lw=0.6, alpha=0.6)
for nm, c, ls in [("avg_dev Electronegativity", RED, "-"), ("mean Row", BLUE, "--"), ("mean Electronegativity", GREEN, "-."),
                  ("maximum Electronegativity", PURPLE, ":")]:
    j = np.where(names == nm)[0][0]; ax.plot(lams, rc[j], color=c, lw=2.4, ls=ls, label=nm)
ax.set_xscale("log"); ax.set_xlabel(r"Ridge $\lambda$"); ax.set_ylabel("Coefficient (eV per s.d.)")
ax.set_ylim(-3, 3); ax.legend(frameon=False, fontsize=12.5, loc="lower right")
panel_label(ax, "a")
ax = axs[0, 1]
cl = S["coefs_l"]
for j in range(cl.shape[0]):
    ax.plot(alphas, cl[j], color=GRAY, lw=0.6, alpha=0.6)
for nm, c, ls in [("avg_dev Electronegativity", RED, "-"), ("mean Row", BLUE, "--"), ("mean Electronegativity", GREEN, "-."),
                  ("maximum Electronegativity", PURPLE, ":")]:
    j = np.where(names == nm)[0][0]; ax.plot(alphas, cl[j], color=c, lw=2.4, ls=ls, label=nm)
ax.axvline(float(S["a1se"]), color=NAVY, ls="--", lw=1.3)
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"LASSO $\alpha$"); ax.set_ylabel("Coefficient (eV per s.d.)")
ax.set_ylim(-1.5, 1.5)
panel_label(ax, "b")
ax = axs[1, 0]
las, ena = S["las"], S["ena"]
ax.errorbar(alphas, las[:, 0], yerr=las[:, 1], color=NAVY, marker="o", ms=5, lw=1.8, capsize=2, label="LASSO")
ax.errorbar(alphas * 1.04, ena[:, 0], yerr=ena[:, 1], color=GOLD, marker="s", ms=5, lw=1.8, ls="--", capsize=2, label="Elastic net (0.5)")
ax.axvline(float(S["a1se"]), color=NAVY, ls="--", lw=1.3)
ax2 = ax.twinx(); ax2.plot(alphas, S["nnz_l"], color=RED, ls=":", lw=2.2); ax2.set_ylabel("Nonzero features (LASSO)", color=RED)
ax2.tick_params(axis="y", colors=RED); ax2.spines["right"].set_visible(True)
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("Grouped CV MAE (eV)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.33, 1.0), fontsize=14)
panel_label(ax, "c")
ax = axs[1, 1]
sl, se = S["sel_l"], S["sel_e"]; o = np.argsort(-(sl + se))[:18][::-1]
yy = np.arange(len(o))
ax.barh(yy + 0.2, sl[o], 0.4, color=NAVY, label="LASSO")
ax.barh(yy - 0.2, se[o], 0.4, color=GOLD, hatch="//", edgecolor="white", label="Elastic net")
ax.set_yticks(yy); ax.set_yticklabels(names[o], fontsize=12)
ax.set_xlabel("Selection frequency (50 bootstraps)"); ax.set_xlim(0, 1.05)
ax.legend(frameon=False, fontsize=13, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
panel_label(ax, "d", x=-0.62)
fig.savefig(os.path.join(HERE, "Fig6.3.2_reg_paths.png")); plt.close(fig)

# ---------- Fig 6.3.3: KRR and GP ----------
K = np.load(os.path.join(LAB, "63_krr_gp_results.npz"))
y = K["y"]
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
ax = axs[0]
ax.scatter(y, K["pr_r"], s=9, color=GRAY, alpha=0.45, label=f"ridge (MAE {np.mean(np.abs(K['pr_r']-y)):.2f})")
ax.scatter(y, K["pr_k"], s=9, color=RED, alpha=0.45, marker="^", label=f"RBF KRR (MAE {np.mean(np.abs(K['pr_k']-y)):.2f})")
ax.plot([0, 12], [0, 12], color=NAVY, lw=1.4, ls="--")
ax.set_xlim(0, 12); ax.set_ylim(-2, 12); ax.set_xlabel("Experimental gap (eV)"); ax.set_ylabel("Predicted gap (eV)")
ax.legend(frameon=False, fontsize=13, loc="upper left", markerscale=2.5)
panel_label(ax, "a")
ax = axs[1]
e = np.abs(K["pl_k"] - y)
ax.scatter(K["kmax"], e, s=9, color=BLUE, alpha=0.5)
bins = np.linspace(0, 1, 11); mid = 0.5 * (bins[1:] + bins[:-1])
med = [np.mean(e[(K["kmax"] >= a) & (K["kmax"] < b)]) if np.any((K["kmax"] >= a) & (K["kmax"] < b)) else np.nan
       for a, b in zip(bins[:-1], bins[1:])]
ax.plot(mid, med, color=RED, lw=2.6, marker="o", label="bin mean")
ax.set_xlabel(r"max$_j\,k(\mathbf{x},\mathbf{x}_j)$ (leave-cluster-out)"); ax.set_ylabel("KRR |error| (eV)")
ax.set_ylim(0, 6); ax.legend(frameon=False, fontsize=14)
panel_label(ax, "b")
ax = axs[2]
sd, err = K["sd"], np.abs(K["mu"] - K["yte"])
ax.scatter(sd, err, s=12, color=PURPLE, alpha=0.5)
xs = np.linspace(0, sd.max() * 1.05, 50)
ax.plot(xs, xs, color=NAVY, lw=1.6, ls="--", label=r"$|e| = \sigma$")
ax.plot(xs, 1.96 * xs, color=RED, lw=1.6, ls=":", label=r"$|e| = 1.96\sigma$")
ax.set_xlabel(r"GP predictive std $\sigma$ (eV)"); ax.set_ylabel("|error| (eV)")
ax.set_xlim(0, sd.max() * 1.05); ax.set_ylim(0, 6); ax.legend(frameon=False, fontsize=14, loc="upper left")
panel_label(ax, "c")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.3.3_krr_gp.png")); plt.close(fig)
print("done")
