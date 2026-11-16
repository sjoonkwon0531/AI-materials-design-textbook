"""Figures for Sec. 7.6: Fig7.6.1 (MLP vs LightGBM, ensemble spread), Fig7.6.2 (e3nn block + checks),
Fig7.6.3 (DP vs DDP schematic). Reads ../labs/76_mlp_bandgap_oof.json and prints the numbers it plots."""
import os, sys, json
import numpy as np, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = os.path.join(HERE, "..", "labs")

# ---------------- Fig 7.6.1 ----------------
d = json.load(open(os.path.join(LABS, "76_mlp_bandgap_oof.json")))
fig, axs = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32))
ax = axs[0]
groups = ["all", "metals", "non-metals"]
models = [("gb", "LightGBM", NAVY, ""), ("single", "MLP (single)", GOLD, "//"), ("mu", "MLP (ensemble of 4)", RED, "..")]
w = 0.13; x = np.arange(3)
for si, (scheme, alpha) in enumerate([("random 5-fold (seed 0)", 1.0), ("group 5-fold (chemsys)", 0.45)]):
    y = np.array(d[scheme]["y"]); masks = [np.ones_like(y, bool), y == 0, y > 0]
    for mi, (k, lab, c, h) in enumerate(models):
        p = np.array(d[scheme][k])
        v = [np.abs(p[m] - y[m]).mean() for m in masks]
        print("Fig7.6.1a %-24s %-20s all %.3f metals %.3f non-metals %.3f" % (scheme, lab, *v))
        ax.bar(x + (mi - 1) * 2 * w + (si - 0.5) * w, v, w, color=c, alpha=alpha, hatch=h, edgecolor="white",
               label=lab if si == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("MAE (eV)")
ax.set_ylim(0, 0.95)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=14, title="solid: random CV,  pale: group CV", title_fontsize=14)
panel_label(ax, "a")
ax = axs[1]
s = d["random 5-fold (seed 0)"]; y = np.array(s["y"]); mu = np.array(s["mu"]); sd = np.array(s["sd"])
err = np.abs(mu - y); q = np.quantile(sd, np.linspace(0, 1, 11)); b = np.clip(np.digitize(sd, q[1:-1]), 0, 9)
ms = np.array([sd[b == i].mean() for i in range(10)]); me = np.array([err[b == i].mean() for i in range(10)])
print("Fig7.6.1b decile mean sd:", np.round(ms, 3)); print("Fig7.6.1b decile MAE  :", np.round(me, 3))
ax.plot(ms, me, "o-", color=RED, lw=2, ms=8, label="observed")
xx = np.linspace(0, ms.max() * 1.05, 50)
ax.plot(xx, np.sqrt(2 / np.pi) * xx, "--", color=GRAY, lw=2, label=r"Gaussian: $\sqrt{2/\pi}\,\sigma$")
ax.set_xlabel(r"ensemble spread $\sigma$ (eV)"); ax.set_ylabel(r"mean $|\mathrm{error}|$ (eV)")
ax.legend(frameon=False, loc="upper left"); panel_label(ax, "b")
fig.savefig(os.path.join(HERE, "Fig7.6.1_mlp_vs_gbdt.png")); plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=NAVY, fs=15, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, a, b, c=NAVY, ls="-", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------- Fig 7.6.2: the interaction block and the symmetry checks ----------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.25))
ax = axs[0]; ax.set_xlim(-0.1, 10); ax.set_ylim(0, 7); ax.axis("off")
box(ax, 1.5, 6.0, 2.8, 0.8, r"$h_j$: 16x0e+8x1o+8x1e", "#CADCFC", fs=13)
box(ax, 4.35, 6.0, 2.4, 0.8, r"$Y^{(l\leq1)}(\hat r_{ij})$", "#FDEBD0", fs=14)
box(ax, 7.75, 6.0, 3.3, 0.8, r"Bessel$(|r_{ij}|)$ x envelope", "#FDEBD0", fs=13)
box(ax, 7.75, 4.6, 2.2, 0.8, "radial MLP", "#FDEBD0")
box(ax, 3.6, 3.3, 3.4, 0.9, "tensor product (CG)\nweights per edge", "#D5F5E3", fs=14)
box(ax, 3.6, 1.95, 2.6, 0.75, r"$\sum_{j}$ (scatter-add)", "#D5F5E3", fs=14)
box(ax, 1.0, 1.95, 1.6, 0.75, r"Linear($h_i$)", "#CADCFC", fs=14)
box(ax, 2.5, 0.6, 4.4, 0.75, "add, then Gate (SiLU / sigmoid)", "#E8DAEF", fs=13)
arrow(ax, (1.5, 5.6), (3.0, 3.75)); arrow(ax, (4.3, 5.6), (3.8, 3.75)); arrow(ax, (7.75, 5.6), (7.75, 5.0))
arrow(ax, (6.65, 4.6), (5.3, 3.55)); arrow(ax, (3.6, 2.85), (3.6, 2.33)); arrow(ax, (3.2, 1.57), (2.7, 0.98))
arrow(ax, (1.0, 1.57), (1.7, 0.98))
ax.text(8.0, 2.4, "only invariant inputs\n(distance) set the\nweights", fontsize=13, color=GRAY, ha="center")
ax.text(5.3, 1.0, "equivariant:\nthe direction enters\nonly through $Y$", fontsize=13, color=GRAY, ha="left", va="center")
panel_label(ax, "a", x=-0.02)
ax = axs[1]
vals = {}
for line in open(os.path.join(LABS, "76_e3nn_block.out")):
    if line.startswith("hidden features"): vals["hidden h\n(float64)"] = float(line.split("=")[1].split()[0])
    elif line.startswith("energy:"): vals["energy\n(float64)"] = float(line.split("=")[1].split()[0])
    elif line.startswith("forces:"): vals["forces\n(float64)"] = float(line.split("=")[1].split()[0])
    elif line.startswith("float32"):
        p = line.split("="); vals["forces\n(float32)"] = float(p[2].split()[0])
    elif line.startswith("bug"):
        vals["energy, bug\n(xyz as 4x0e)"] = float(line.split("=")[1].split()[0])
print("Fig7.6.2b:", vals)
cols = [RED if "bug" in k else (BLUE if "32" in k else NAVY) for k in vals]
ax.bar(range(len(vals)), list(vals.values()), color=cols, edgecolor="white")
ax.set_yscale("log"); ax.set_ylim(1e-17, 1); ax.set_xticks(range(len(vals)))
ax.set_xticklabels(list(vals.keys()), fontsize=13)
ax.set_ylabel("max equivariance error under rotation")
panel_label(ax, "b")
fig.savefig(os.path.join(HERE, "Fig7.6.2_e3nn_block.png")); plt.close(fig)

# ---------------- Fig 7.6.3: DP vs DDP ----------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(wspace=0.08))
ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
box(ax, 5, 7.2, 7.4, 0.8, "1 process (Python), GPU 0 holds master weights", "#CADCFC", fs=14)
for i, xg in enumerate([2.0, 5.0, 8.0]):
    box(ax, xg, 4.6, 2.2, 1.4, f"GPU {i}\nreplica (copied\nevery step)", "#FDEBD0", fs=13)
    arrow(ax, (5, 6.8), (xg, 5.3), c=BLUE)
    arrow(ax, (xg, 3.9), (5, 2.2), c=RED)
box(ax, 5, 1.7, 5.0, 0.9, "gather outputs/grads on GPU 0\nreduce, update, scatter again", "#FADBD8", fs=14)
ax.text(5, 0.4, "threads share one interpreter (GIL); GPU 0 is the bottleneck", ha="center", fontsize=13, color=GRAY)
panel_label(ax, "a", x=0.0)
ax = axs[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
for i, xg in enumerate([1.8, 5.0, 8.2]):
    box(ax, xg, 6.4, 2.6, 1.6, f"process {i}\nfull replica\nown data shard", "#D5F5E3", fs=13)
    box(ax, xg, 4.1, 2.4, 0.8, "backward", "#FDEBD0", fs=14)
    arrow(ax, (xg, 5.6), (xg, 4.5)); arrow(ax, (xg, 3.7), (xg, 2.85))
    box(ax, xg, 1.0, 2.4, 0.8, "same update", "#D5F5E3", fs=14)
    arrow(ax, (xg, 2.05), (xg, 1.4))
box(ax, 5, 2.45, 9.2, 0.8, "all-reduce (average) gradient buckets, overlapped with backward", "#FADBD8", fs=14)
panel_label(ax, "b", x=0.0)
fig.savefig(os.path.join(HERE, "Fig7.6.3_dp_ddp.png")); plt.close(fig)
print("done")
