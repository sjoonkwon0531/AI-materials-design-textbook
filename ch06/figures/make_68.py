"""Figures for Section 6.8 (perovskite case study). Reads labs/68_out.npz written by labs/68_perovskite.py."""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.join(HERE, "..", "labs")
D = np.load(os.path.join(LAB, "68_out.npz"), allow_pickle=True)
t, tau, y, P, X = D["t"], D["tau"], D["y"], D["P"], D["X"]
rng = np.random.default_rng(1)

# ---------- Fig 6.8.1: t vs tau as classifiers ----------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
for ax, v, lo, hi, lab in ((axs[0], t, 0.825, 1.059, r"Goldschmidt $t$"), (axs[1], np.log10(tau), None, np.log10(4.18), r"$\log_{10}\tau$")):
    for k, (cls, col, mk, nm) in enumerate(((1, NAVY, "o", "perovskite"), (0, RED, "s", "non-perovskite"))):
        m = y == cls
        ax.scatter(v[m], k + 0.28 * (rng.random(m.sum()) - 0.5), s=14, color=col, marker=mk, alpha=0.55, label=nm, lw=0)
    if lo is not None: ax.axvspan(lo, hi, color=GOLD, alpha=0.25, lw=0)
    else: ax.axvspan(v.min() - 0.05, hi, color=GOLD, alpha=0.25, lw=0)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["perovskite", "non-\nperovskite"]); ax.set_xlabel(lab); ax.set_ylim(-0.5, 1.5)
axs[0].set_xlim(0.68, 1.2); axs[1].set_xlim(0.0, 1.4)
an = ["O", "F", "Cl", "Br", "I"]
acc_t = [np.mean((((t > 0.825) & (t < 1.059)).astype(int) == y)[X == a]) for a in an]
acc_T = [np.mean(((tau < 4.18).astype(int) == y)[X == a]) for a in an]
xx = np.arange(5)
axs[2].bar(xx - 0.2, acc_t, 0.38, color=GRAY, label=r"$t$", hatch="//", edgecolor="white")
axs[2].bar(xx + 0.2, acc_T, 0.38, color=NAVY, label=r"$\tau$")
axs[2].set_xticks(xx); axs[2].set_xticklabels(an); axs[2].set_ylabel("Accuracy"); axs[2].set_ylim(0, 1.2)
axs[2].set_xlabel("Anion X"); axs[2].legend(loc="upper center", frameon=False, ncol=2)
for a, l in zip(axs, "abc"): panel_label(a, l, x=-0.2 if l != "c" else -0.14)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.8.1_t_vs_tau.png")); plt.close(fig)

# ---------- Fig 6.8.2: Platt curve, learning curves, extrapolation ----------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axs[0]
g = np.linspace(1.5, 8, 300); pa, pb = np.polyfit(tau, np.log(P / (1 - P)), 1)
ax.plot(g, 1 / (1 + np.exp(-(pa * g + pb))), color=NAVY, lw=2.2, label=r"Platt $P(\tau)$")
edges = np.array([1.0, 3.0, 3.5, 3.8, 4.0, 4.2, 4.4, 4.7, 5.2, 6.0, 8.0])
cen, frac, nn = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (tau >= lo) & (tau < hi)
    if m.sum() > 4: cen.append(np.median(tau[m])); frac.append(y[m].mean()); nn.append(m.sum())
ax.plot(cen, frac, "s", color=RED, ms=8, label="observed fraction")
ax.axvline(4.18, color=GRAY, ls="--", lw=1.4)
ax.set_xlabel(r"$\tau$"); ax.set_ylabel("P(perovskite)"); ax.set_xlim(1.5, 8); ax.legend(frameon=False, loc="upper right")
ax = axs[1]
sz = D["sizes"]
for arr, col, mk, ls, nm in ((D["lc_tau"], NAVY, "o", "-", r"$\tau$ stump (1 threshold)"), (D["lc_raw"], BLUE, "s", "--", "RF on (r, n)"),
                             (D["lc_mag"], RED, "^", ":", "RF on Magpie")):
    ax.errorbar(sz, arr[:, 0], yerr=arr[:, 1], color=col, marker=mk, ls=ls, lw=2, ms=7, capsize=3, label=nm)
ax.set_xscale("log"); ax.set_xticks(sz); ax.set_xticklabels([str(v) for v in sz]); ax.minorticks_off()
ax.set_xlabel("Training compounds"); ax.set_ylabel("Test accuracy"); ax.set_ylim(0.6, 1.0)
ax.legend(frameon=False, loc="lower right")
ax = axs[2]
vals = {"oxides to halides": (0.913, 0.522, 0.609), "halides to oxides": (0.919, 0.339, 0.642)}  # from labs/68_perovskite.out
xx = np.arange(2)
for k, (col, nm, h) in enumerate(((NAVY, r"$\tau<4.18$", None), (BLUE, "RF on (r, n)", "//"), (RED, "RF on Magpie", ".."))):
    ax.bar(xx + (k - 1) * 0.27, [vals[v][k] for v in vals], 0.26, color=col, label=nm, hatch=h, edgecolor="white")
ax.axhline(0.5, color=GRAY, ls=":", lw=1.2)
ax.set_xticks(xx); ax.set_xticklabels(["oxides to\nhalides", "halides to\noxides"]); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.18)
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=13)
for a, l in zip(axs, "abc"): panel_label(a, l)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.8.2_platt_lc_extrap.png")); plt.close(fig)
print("observed-fraction bins:", list(zip(np.round(cen, 2), np.round(frac, 2), nn)))
print("Platt line refit from P: a=%.3f b=%.3f" % (pa, pb))
