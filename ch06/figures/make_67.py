"""Figures for Section 6.7. Reads labs/67_cv_results.json, labs/67_pipeline_out.npz, labs/67_soap_emt.npz."""
import sys, os, json, logging
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np, pandas as pd
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.join(HERE, "..", "labs")
R = json.load(open(os.path.join(LAB, "67_cv_results.json")))
df = pd.read_parquet(os.path.join(LAB, "67_features_clean.parquet")); y = df.target.values
# ---------- Fig 6.7.1: validation schemes ----------
fig, axs = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
names = ["random\n5-fold", "group\n(chem. system)", "LOCO\nk = 10", "LOCO\nk = 5"]
fold = [np.ravel(R["random"]), R["group"], R["loco10"], R["loco5"]]
# pooled MAE values printed by labs/67_cv_leakage.py
pooled = [np.mean([0.359, 0.356, 0.360]), 0.376, 0.568, 0.688]
ax = axs[0]; xx = np.arange(4)
ax.bar(xx, pooled, 0.6, color=[NAVY, BLUE, GOLD, RED])
for i, f in enumerate(fold):
    ax.scatter(np.full(len(f), i) + 0.18 * (np.random.default_rng(i).random(len(f)) - 0.5), f, color="k", s=16, zorder=3)
ax.axhline(1.143, color=GRAY, ls="--", lw=1.4); ax.text(3.35, 1.16, "predict mean", ha="right", va="bottom", fontsize=14, color=GRAY)
ax.set_xticks(xx); ax.set_xticklabels(names, fontsize=14); ax.set_ylabel("MAE (eV)"); ax.set_ylim(0, 1.6)
ax = axs[1]
lab = np.array(R["lab5"]); fam = np.array(R["fam"])
fams = ["oxide", "chalcogenide", "halide", "pnictide", "other"]; cols = [RED, GOLD, GREEN, PURPLE, GRAY]
hat = [None, "//", "..", "xx", None]
bottom = np.zeros(5)
for f, c, h in zip(fams, cols, hat):
    v = np.array([np.mean(fam[lab == k] == f) for k in range(5)])
    ax.bar(np.arange(5), v, 0.6, bottom=bottom, color=c, label=f, hatch=h, edgecolor="white"); bottom += v
ax.set_ylabel("Family fraction in cluster"); ax.set_xticks(range(5)); ax.set_xticklabels([f"C{k}" for k in range(5)])
ax.set_xlabel("Held-out cluster"); ax.set_ylim(0, 1.0)
ax2 = ax.twinx(); ax2.plot(range(5), R["loco5"], "D-", color=NAVY, ms=9, lw=2, label="LOCO MAE")
ax2.set_ylabel("MAE when held out (eV)"); ax2.set_ylim(0, 1.6); ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=12, ncol=1, loc="center left", bbox_to_anchor=(1.13, 0.5))
panel_label(axs[0], "a"); panel_label(axs[1], "b", x=-0.12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.7.1_cv_leakage.png")); plt.close(fig)

# ---------- Fig 6.7.2: pipeline diagnostics ----------
D = np.load(os.path.join(LAB, "67_pipeline_out.npz"), allow_pickle=True)
fig, axs = plt.subplots(1, 3, figsize=(17, 5.0))
ax = axs[0]
yt, yd, y2 = D["y"], D["yd"], D["y2"]
ax.scatter(yt, yd, s=12, color=GRAY, alpha=0.5, marker="s", label="direct", lw=0)
ax.scatter(yt, y2, s=12, color=NAVY, alpha=0.6, label="two-stage", lw=0)
ax.plot([0, 9], [0, 9], color=RED, lw=1.4); ax.set_xlim(-0.2, 9); ax.set_ylim(-0.2, 9)
ax.set_xlabel("Measured gap (eV)"); ax.set_ylabel("Predicted gap (eV)"); ax.legend(frameon=False, loc="upper left", markerscale=2)
ax = axs[1]
fams = ["oxide", "chalcogenide", "halide", "pnictide", "other"]; fm = D["fam"]
md = [np.mean(np.abs(yt - yd)[fm == f]) for f in fams]; m2 = [np.mean(np.abs(yt - y2)[fm == f]) for f in fams]
xx = np.arange(5)
ax.bar(xx - 0.2, md, 0.38, color=GRAY, hatch="//", edgecolor="white", label="direct")
ax.bar(xx + 0.2, m2, 0.38, color=NAVY, label="two-stage")
ax.set_xticks(xx); ax.set_xticklabels(["oxide", "chalco-\ngenide", "halide", "pnictide", "other"], fontsize=14)
ax.set_ylabel("Test MAE (eV)"); ax.legend(frameon=False); ax.set_ylim(0, 1.0)
ax = axs[2]
lv = np.array(D["levels"])
ax.plot([0.4, 1], [0.4, 1], color=GRAY, ls=":", lw=1.4)
ax.plot(lv, D["obs_raw"], "o--", color=GOLD, ms=8, lw=2, label=r"RF spread, Gaussian $\sigma$")
ax.plot(lv, D["obs_cal"], "s-", color=NAVY, ms=8, lw=2, label="RF spread, conformal")
cov = np.mean((D["yti"] >= D["q05"]) & (D["yti"] <= D["q95"]))
ax.plot([0.9], [cov], "^", color=RED, ms=12, label="quantile LightGBM")
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage"); ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
ax.legend(frameon=False, loc="lower right", fontsize=13)
for a, l in zip(axs, "abc"): panel_label(a, l)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.7.2_pipeline.png")); plt.close(fig)
print("quantile coverage check:", round(cov, 3))

# ---------- Fig 6.7.3: SOAP lesson ----------
S = np.load(os.path.join(LAB, "67_soap_emt.npz"), allow_pickle=True)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.0))
ax = axs[0]; ys = S["y"]; kd = S["kind"]
for k, c, mk in (("ordered", BLUE, "o"), ("random", GOLD, "s"), ("segregated", RED, "^")):
    m = kd == k
    ax.scatter(ys[m], S["oof_comp"][m], s=22, color=c, marker=mk, alpha=0.75, lw=0, label=k + " (composition)")
ax.scatter(ys, S["oof_res"], s=10, color=NAVY, marker="x", lw=1, label="SOAP, centre-resolved")
ax.plot([0, 110], [0, 110], color=GRAY, lw=1.2)
ax.set_xlabel("EMT energy (meV/atom)"); ax.set_ylabel("Predicted (meV/atom)"); ax.legend(frameon=False, fontsize=12, loc="upper left")
ax = axs[1]; lc = S["lc"]
ax.errorbar(lc[:, 0], lc[:, 1], yerr=lc[:, 2], color=RED, marker="s", ls="--", lw=2, capsize=3, label="composition only")
ax.errorbar(lc[:, 0], lc[:, 3], yerr=lc[:, 4], color=NAVY, marker="o", lw=2, capsize=3, label="SOAP, centre-resolved")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(lc[:, 0]); ax.set_xticklabels([str(int(v)) for v in lc[:, 0]]); ax.minorticks_off()
ax.set_xlabel("Training cells"); ax.set_ylabel("Test MAE (meV/atom)"); ax.legend(frameon=False, loc="lower left")
for a, l in zip(axs, "ab"): panel_label(a, l)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.7.3_soap_pooling.png")); plt.close(fig)
