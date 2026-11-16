"""Figures for Sec. 6.6 (LCA). Reads labs/66_mclca.json and labs/66_mllca.json."""
import sys, json, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.join(HERE, "..", "labs")

# ---------- Fig 6.6.1: ISO 14040 framework ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
def box(x, y, w, h, txt, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=NAVY, lw=1.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=17)
box(0.3, 3.85, 5.6, 1.0, "Goal and scope definition\n(functional unit, system boundary)", ICE)
box(0.3, 2.15, 5.6, 1.0, "Inventory analysis (LCI)\n(input and output flows per FU)", ICE)
box(0.3, 0.45, 5.6, 1.0, "Impact assessment (LCIA)\n(characterization, e.g. GWP100)", ICE)
box(7.0, 0.45, 2.7, 4.4, "Interpretation\n(contribution,\nuncertainty,\nsensitivity)", "#FDF1D6")
kw = dict(arrowstyle="<->", color=NAVY, lw=1.6, mutation_scale=18)
for y0, y1 in [(3.85, 3.15), (2.15, 1.45)]:
    ax.annotate("", xy=(3.1, y1), xytext=(3.1, y0), arrowprops=kw)
for yc in (4.35, 2.65, 0.95):
    ax.annotate("", xy=(6.92, yc), xytext=(6.0, yc), arrowprops=kw)
fig.savefig(os.path.join(HERE, "Fig6.6.1_lca_framework.png")); plt.close(fig)

# ---------- Fig 6.6.2: Monte Carlo LCA ----------
d = json.load(open(os.path.join(LAB, "66_mclca.json")))
A, B, D, D2 = map(np.array, (d["totA"], d["totB"], d["D"], d["D2"]))
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
ax = axs[0, 0]; bins = np.linspace(5, 60, 90)
ax.hist(A, bins, density=True, histtype="stepfilled", alpha=0.35, color=RED, label="route A")
ax.hist(B, bins, density=True, histtype="step", lw=2.2, color=BLUE, ls="--", label="route B")
for r, c, ls in (("A", RED, "-"), ("B", BLUE, "--")):
    ax.axvline(sum(d["det"][r].values()), color=c, ls=ls, lw=1.6)
ax.set_xlabel("GWP100 (kg CO$_2$e per kg X)"); ax.set_ylabel("Probability density"); ax.legend(frameon=False)
panel_label(ax, "a")
ax = axs[0, 1]; S = d["sens"][:6][::-1]
NM={"inv elec":"electricity use","EF elec_CO2":"grid CO$_2$ factor","EF P2":"P2 factor","EF P1":"P1 factor","EF elec_CH4":"grid CH$_4$ factor","inv gas":"heat use","EF gas_CO2":"gas CO$_2$ factor","inv P1":"P1 amount","inv P2":"P2 amount","inv proc_CO2":"process CO$_2$"}
ax.barh([NM.get(s[0],s[0]) for s in S], [100 * s[2] for s in S], color=NAVY)
ax.set_xlabel("Share of rank variance, route A (%)"); panel_label(ax, "b", x=-0.36)
ax = axs[1, 0]; bins = np.linspace(-25, 40, 90)
ax.hist(D, bins, density=True, histtype="stepfilled", alpha=0.35, color=GREEN, label="shared background")
ax.hist(D2, bins, density=True, histtype="step", lw=2.2, color=PURPLE, ls="--", label="independent (wrong)")
ax.axvline(0, color=GRAY, lw=1.2); ax.set_xlabel("GWP(A) $-$ GWP(B) (kg CO$_2$e)")
ax.set_ylabel("Probability density"); ax.legend(frameon=False, loc="upper right"); panel_label(ax, "c")
ax = axs[1, 1]; bins = np.linspace(0, 40, 90)
ax.hist(np.array(d["totlA"]), bins, density=True, histtype="stepfilled", alpha=0.35, color=RED, label="route A")
ax.hist(np.array(d["totlB"]), bins, density=True, histtype="step", lw=2.2, color=BLUE, ls="--", label="route B")
ax.set_xlabel("GWP100, low-carbon grid (kg CO$_2$e)"); ax.set_ylabel("Probability density")
ax.legend(frameon=False); panel_label(ax, "d")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.6.2_mc_lca.png")); plt.close(fig)

# ---------- Fig 6.6.3: ML-LCA extrapolation ----------
m = json.load(open(os.path.join(LAB, "66_mllca.json")))
yin, yout = np.array(m["yin"]), np.array(m["yout"])
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
ax = axs[0]
Xtr, Xout = np.array(m["Xtr"]), np.array(m["Xout"])
jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(Xtr))
ax.scatter(Xtr[:, 0] + jit, Xtr[:, 1] + np.log(100), s=12, color=NAVY, alpha=0.6, label="train")
jit = np.random.default_rng(1).uniform(-0.25, 0.25, len(Xout))
ax.scatter(Xout[:, 0] + jit, Xout[:, 1] + np.log(100), s=12, color=GOLD, marker="^", alpha=0.7, label="out of domain")
ax.set_xlabel("Synthesis steps"); ax.set_ylabel("ln(MW / g mol$^{-1}$)"); ax.legend(frameon=False, loc="upper left")
panel_label(ax, "a")
ax = axs[1]
for k, c, mk in (("RF", RED, "o"), ("linear (ln GWP)", BLUE, "s")):
    ax.scatter(yin, m["pin"][k], s=10, color=c, marker=mk, alpha=0.35)
    ax.scatter(yout, m["pout"][k], s=14, color=c, marker=mk, alpha=0.8, label=("RF" if k == "RF" else "linear"))
lim = [-1.5, 4.5]; ax.plot(lim, lim, color=GRAY, ls=":", lw=1.5)
ax.axvline(max(m["ytr"]), color=GRAY, ls="--", lw=1.2)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("True ln GWP"); ax.set_ylabel("Predicted ln GWP")
ax.legend(frameon=False, loc="upper left"); panel_label(ax, "b")
ax = axs[2]
ax.scatter(m["d_in"], m["e_in"], s=12, color=NAVY, alpha=0.5, label="in domain")
ax.scatter(m["d_out"], m["e_out"], s=14, color=GOLD, marker="^", alpha=0.7, label="out of domain")
ax.axvline(m["thr"], color=RED, ls="--", lw=1.6)
ax.set_xlabel("Mean 5-NN distance to training set"); ax.set_ylabel("|error| of RF (ln units)")
ax.legend(frameon=False, loc="lower right"); panel_label(ax, "c")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.6.3_mllca_extrapolation.png")); plt.close(fig)
print("done")
