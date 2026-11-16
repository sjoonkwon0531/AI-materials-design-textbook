"""Figures for Sec.9.3 (neural operators).  Usage: python make_93.py [1 2 3 4 5]
Fig9.3.1 schematic (FNO layer / DeepONet), Fig9.3.2 FNO on Burgers, Fig9.3.3 DeepONet vs FNO,
Fig9.3.4 PINO, Fig9.3.5 KS rollout stability.  Data: labs/93_*.json written by the lab scripts."""
import os, sys, json
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); from bookstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = os.path.join(HERE, "..", "labs")
J = lambda name: json.load(open(os.path.join(LABS, name)))
which = sys.argv[1:] or ["1", "2", "3", "4", "5"]


def box(ax, x, y, w, h, text, fc=ICE, ec=NAVY, fs=15, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, **kw)


def arrow(ax, p, q, color=NAVY, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=1.8, color=color, **kw))


# ------------------------------------------------------------------ Fig 9.3.1 schematic
if "1" in which:
    fig, axs = plt.subplots(2, 1, figsize=(15, 9.2), gridspec_kw=dict(height_ratios=[1.1, 1], hspace=0.12))
    ax = axs[0]; ax.set_xlim(0, 15); ax.set_ylim(0, 5.2); ax.axis("off"); panel_label(ax, "a", x=-0.02, y=0.93)
    box(ax, 0.2, 2.0, 1.5, 1.2, "$a(x)$", fc="w")
    box(ax, 2.3, 2.0, 1.5, 1.2, "lift $P$\n(pointwise)")
    ax.add_patch(FancyBboxPatch((4.5, 0.35), 6.6, 4.5, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F7F7F7", ec=GRAY, lw=1.4, ls="--"))
    ax.text(7.8, 4.55, r"Fourier layer  ($\times L$)", ha="center", fontsize=16, color=GRAY)
    box(ax, 4.9, 3.0, 1.3, 1.0, r"$\mathcal{F}$", fc="w")
    box(ax, 6.7, 3.0, 2.2, 1.0, r"$R(k)\cdot$,  $|k|\leq k_{\max}$", fc=GOLD, ec=NAVY, fs=14)
    box(ax, 9.4, 3.0, 1.3, 1.0, r"$\mathcal{F}^{-1}$", fc="w")
    box(ax, 6.7, 0.9, 2.2, 1.0, "$W$  (pointwise)", fc=ICE, fs=14)
    ax.text(11.75, 2.6, r"$+$", fontsize=26, ha="center", va="center")
    box(ax, 12.3, 2.0, 0.9, 1.2, r"$\sigma$", fc="w")
    box(ax, 13.6, 2.0, 1.25, 1.2, "project\n$Q$")
    arrow(ax, (1.75, 2.6), (2.25, 2.6)); arrow(ax, (3.85, 2.6), (4.45, 2.6))
    arrow(ax, (4.45, 2.6), (4.9, 3.4), connectionstyle="arc3,rad=-0.2"); arrow(ax, (6.25, 3.5), (6.65, 3.5)); arrow(ax, (8.95, 3.5), (9.35, 3.5))
    arrow(ax, (10.75, 3.4), (11.55, 2.85), connectionstyle="arc3,rad=-0.2")
    arrow(ax, (4.45, 2.6), (6.65, 1.4), connectionstyle="arc3,rad=0.2"); arrow(ax, (8.95, 1.4), (11.55, 2.35), connectionstyle="arc3,rad=0.2")
    arrow(ax, (11.95, 2.6), (12.25, 2.6)); arrow(ax, (13.25, 2.6), (13.55, 2.6))
    ax.text(14.95, 2.6, "$u(x)$", fontsize=16, va="center"); ax.set_xlim(0, 15.8)
    ax = axs[1]; ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off"); panel_label(ax, "b", x=-0.02, y=0.93)
    box(ax, 0.2, 2.7, 3.2, 1.2, "sensors\n$a(x_1),\\ldots,a(x_m)$", fc="w", fs=14)
    box(ax, 4.2, 2.7, 2.6, 1.2, "branch net", fc=ICE)
    box(ax, 7.6, 2.7, 2.6, 1.2, "$b_1(a),\\ldots,b_p(a)$", fc=GOLD, fs=14)
    box(ax, 0.2, 0.5, 3.2, 1.2, "query point  $y$", fc="w", fs=14)
    box(ax, 4.2, 0.5, 2.6, 1.2, "trunk net", fc=ICE)
    box(ax, 7.6, 0.5, 2.6, 1.2, "$t_1(y),\\ldots,t_p(y)$", fc=GOLD, fs=14)
    box(ax, 11.2, 1.55, 3.5, 1.3, r"$\sum_k b_k(a)\,t_k(y)\;\approx\;\mathcal{G}(a)(y)$", fc="w", fs=15)
    for yy in (3.3, 1.1):
        arrow(ax, (3.45, yy), (4.15, yy)); arrow(ax, (6.85, yy), (7.55, yy))
    arrow(ax, (10.25, 3.3), (11.6, 2.9), connectionstyle="arc3,rad=-0.15"); arrow(ax, (10.25, 1.1), (11.6, 1.5), connectionstyle="arc3,rad=0.15")
    fig.savefig(os.path.join(HERE, "Fig9.3.1_operator_architectures.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 9.3.2 FNO on Burgers
if "2" in which:
    D = J("93_fno_burgers.json"); A = J("93_fno_ablation.json"); ex = D["example"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10.5), gridspec_kw=dict(hspace=0.36, wspace=0.3))
    ax = axs[0, 0]; x = np.arange(512) / 512
    ax.plot(x, ex["a"], color=GRAY, lw=1.6, ls=":", label="$u_0$")
    ax.plot(x, ex["u"], color="k", lw=2.6, label="solver $u(T)$")
    ax.plot(x, ex["p64"], color=RED, lw=2.0, ls="--", label="FNO: train 64, eval 512")
    ax.plot(x, ex["cnn"], color=BLUE, lw=1.4, ls="-.", label="CNN: train 64, eval 512")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(frameon=False, fontsize=13, loc="best"); panel_label(ax, "a")
    ax = axs[0, 1]; R = D["res"]
    for s, c, m, ls in ((32, GOLD, "o", "-"), (64, RED, "s", "--"), (128, NAVY, "^", "-.")):
        ax.plot(R, D[f"fno_train{s}"], color=c, marker=m, ms=9, lw=2.2, ls=ls, label=f"FNO trained at {s}")
    ax.plot(R, D["cnn_train64"], color=BLUE, marker="D", ms=8, lw=2.0, ls=":", label="CNN trained at 64")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(R); ax.set_xticklabels(R)
    ax.set_xlabel("test resolution (grid points)"); ax.set_ylabel("test relative $L^2$ error"); ax.set_ylim(0.015, 1.3)
    ax.legend(frameon=False, fontsize=13, loc="center right"); panel_label(ax, "b")
    ax = axs[1, 0]; k = np.arange(257)
    ax.semilogy(k[1:160], D["spec_true"][1:160], color="k", lw=2.4, label="solver")
    ax.semilogy(k[1:160], D["spec_pred64"][1:160], color=RED, lw=2.0, ls="--", label="FNO (train 64, eval 512)")
    ax.semilogy(k[1:160], D["spec_err64"][1:160], color=BLUE, lw=1.8, ls="-.", label="error")
    ax.axvline(16, color=GRAY, lw=1.2, ls=":"); ax.axvline(32, color=GRAY, lw=1.2, ls="--")
    ax.text(17, 2.5e-7, "$k_{\\max}$", fontsize=14, color=GRAY); ax.text(33, 2.5e-7, "training Nyquist", fontsize=14, color=GRAY)
    ax.set_xlabel("wavenumber index k"); ax.set_ylabel("rms $|\\hat{u}_k|$"); ax.set_ylim(1e-7, 1); ax.legend(frameon=False, fontsize=13)
    panel_label(ax, "c")
    ax = axs[1, 1]
    ax.plot(A["kmax"], A["kmax_err"], color=RED, marker="s", ms=10, lw=2.2, label="FNO test error")
    ax.plot(A["kmax"], A["kmax_tail"], color="k", marker="o", ms=8, lw=2.0, ls="--", label="solution content above $k_{\\max}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(A["kmax"]); ax.set_xticklabels(A["kmax"])
    ax.set_xlabel("$k_{\\max}$"); ax.set_ylabel("relative $L^2$"); ax.legend(frameon=False, fontsize=13); panel_label(ax, "d")
    fig.savefig(os.path.join(HERE, "Fig9.3.2_fno_burgers.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 9.3.3 DeepONet vs FNO
if "3" in which:
    D = J("93_deeponet.json")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]; p = D["p"][:5]
    ax.plot(p, D["pod_floor_lab"][:5], color=NAVY, marker="o", ms=9, lw=2.4, label="lab frame")
    ax.plot(p, np.array(D["pod_floor_comoving"][:5]) * D["norm_ratio"], color=RED, marker="s", ms=8, lw=2.0, ls="--", label="co-moving frame")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(p); ax.set_xticklabels(p)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5"]); ax.minorticks_off()
    ax.set_xlabel("number of basis functions p"); ax.set_ylabel("POD floor (relative $L^2$)"); ax.legend(frameon=False); panel_label(ax, "a")
    ax = axs[1]; n = D["ntr"]
    ax.plot(n, D["deeponet"], color=NAVY, marker="o", ms=9, lw=2.2, label="DeepONet (p = 64)")
    ax.plot(n, D["pod_deeponet"], color=GREEN, marker="^", ms=9, lw=2.2, ls="-.", label="POD-DeepONet (p = 32)")
    ax.plot(n, D["fno"], color=RED, marker="s", ms=9, lw=2.2, ls="--", label="FNO")
    ax.axhline(D["pod_floor_lab"][4], color=GREEN, lw=1.3, ls=":"); ax.text(n[0], D["pod_floor_lab"][4] * 1.07, "POD floor, p = 32", fontsize=13, color=GREEN)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(n); ax.set_xticklabels(n); ax.minorticks_off()
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4]); ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xlabel("training functions N"); ax.set_ylabel("test relative $L^2$ error"); ax.legend(frameon=False, fontsize=13, loc="center right"); panel_label(ax, "b")
    ax = axs[2]; ex = D["example"]; x = np.arange(64) / 64
    ax.plot(x, ex["a"], color=GRAY, lw=1.6, ls=":", label="$u_0$")
    ax.plot(x, ex["u"], color="k", lw=2.6, label="solver $u(T)$")
    ax.plot(x, ex["don"], color=NAVY, lw=2.0, ls="--", marker="o", ms=4, label="DeepONet (N = 1000)")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(frameon=False, fontsize=13, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.6)
    panel_label(ax, "c")
    fig.savefig(os.path.join(HERE, "Fig9.3.3_deeponet_vs_fno.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 9.3.4 PINO
if "4" in which:
    D = J("93_pino.json")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    keys = ["data25", "both25", "data100", "both100", "pde0"]; labs = ["data\n25", "data+PDE\n25", "data\n100", "data+PDE\n100", "PDE only\n0"]
    cols = [NAVY, RED, NAVY, RED, GOLD]; hat = ["", "//", "", "//", ".."]
    for i, (k, c, h) in enumerate(zip(keys, cols, hat)):
        ax.bar(i, D[k]["err"], color=c, hatch=h, edgecolor="w", width=0.7)
        ax.text(i, D[k]["err"] * 1.03, f"{D[k]['err']:.3f}", ha="center", va="bottom", fontsize=14)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=13); ax.set_ylabel("test relative $L^2$ error")
    ax.set_xlabel("loss and number of labelled trajectories"); ax.set_ylim(0, max(D[k]["err"] for k in keys) * 1.18); panel_label(ax, "a")
    styles = (("data25", NAVY, "-", "o", "from data-only operator"), ("both25", RED, "--", "s", "from data+PDE operator"),
              ("pde0", GOLD, ":", "D", "from PDE-only operator"), ("random", GRAY, "-.", "^", "from random initialisation"))
    for col, ax, yl, lab in ((1, axs[1], "relative $L^2$ error at t = T", "b"), (2, axs[2], "PDE residual (mean square)", "c")):
        for key, c, ls, m, name in styles:
            H = np.array(D["ft_" + key]["hist"])                  # (instances, checkpoints, 3)
            for h in H: ax.plot(h[:, 0], h[:, col], color=c, lw=0.8, alpha=0.35)
            ax.plot(H[0, :, 0], np.exp(np.log(H[:, :, col]).mean(0)), color=c, ls=ls, marker=m, ms=7, lw=2.4, label=name)
        ax.set_yscale("log"); ax.set_xlabel("fine-tuning step"); ax.set_ylabel(yl); panel_label(ax, lab)
    axs[1].legend(frameon=False, fontsize=13)
    fig.savefig(os.path.join(HERE, "Fig9.3.4_pino.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 9.3.5 KS rollout
if "5" in which:
    D = J("93_rollout_eval.json")
    fig = plt.figure(figsize=(18.5, 10.8)); gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.36)
    L = 64.0
    for i, (key, lab) in enumerate((("example_true", "a"), ("example_onestep", "b"))):
        ax = fig.add_subplot(gs[0, i]); U = np.array(D[key])
        im = ax.imshow(U.T, origin="lower", aspect="auto", extent=[0, U.shape[0] - 1, 0, L], cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_xlabel("time t"); ax.set_ylabel("x"); ax.spines[:].set_visible(True); panel_label(ax, lab)
        ax.text(0.03, 0.93, "solver" if i == 0 else "FNO rollout (one-step training)", transform=ax.transAxes, fontsize=14,
                bbox=dict(fc="w", ec="none", alpha=0.85), va="top")
    cax = fig.axes[1].inset_axes([1.02, 0.0, 0.035, 1.0]); cb = fig.colorbar(im, cax=cax); cax.set_title("u", fontsize=16, pad=6)
    sty = {"onestep": (NAVY, "-", "one-step"), "noise": (GOLD, ":", "noise injection"), "push": (GREEN, "-.", "pushforward"),
           "unroll": (RED, "--", "unrolled (K = 4)")}
    ax = fig.add_subplot(gs[0, 2]); t = np.arange(401)
    for k, (c, ls, name) in sty.items(): ax.plot(t[:161], np.array(D[k]["corr"])[:161], color=c, ls=ls, lw=2.4, label=name)
    ax.axhline(0.9, color=GRAY, lw=1.2, ls=":"); ax.set_xlabel("time t"); ax.set_ylabel("correlation with solver"); ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=13, loc="lower left"); panel_label(ax, "c")
    ax = fig.add_subplot(gs[1, 0]); k = np.array(D["k"]); q = 2 * np.pi * k / L
    ax.semilogy(q, D["gain_solver"], color="k", lw=2.6, label="solver")
    ax.semilogy(q, np.maximum(D["gain_linear_exact"], 1e-9), color=GRAY, lw=1.4, ls=":", label=r"$e^{(q^2-q^4)\Delta t}$")
    for kk in ("onestep", "unroll"): ax.semilogy(q, D["gain_" + kk], color=sty[kk][0], ls=sty[kk][1], lw=2.2, label=sty[kk][2])
    ax.axvline(2 * np.pi * 24 / L, color=GRAY, lw=1.2, ls="--"); ax.text(2 * np.pi * 24 / L + 0.08, 2e-6, "$k_{\\max}$", color=GRAY, fontsize=14)
    ax.axhline(1, color=GRAY, lw=0.8); ax.set_ylim(1e-6, 5); ax.set_xlabel("wavenumber q"); ax.set_ylabel("per-step gain $|g(q)|$")
    ax.legend(frameon=False, fontsize=13, loc="upper right"); panel_label(ax, "d")
    ax = fig.add_subplot(gs[1, 1]); kk = np.arange(65); qq = 2 * np.pi * kk / L
    ax.semilogy(qq[1:], np.maximum(D["spec_true_med"][1:], 1e-12), color="k", lw=2.6, label="solver")
    for name in ("onestep", "unroll"):
        ax.semilogy(qq[1:], np.array(D[name]["spec_med"])[1:], color=sty[name][0], ls=sty[name][1], lw=2.2, label=sty[name][2] + " (median)")
    ax.semilogy(qq[1:], np.array(D["onestep"]["spec_worst"])[1:], color=NAVY, lw=1.0, alpha=0.6, label="one-step (worst trajectory)")
    ax.set_xlabel("wavenumber q"); ax.set_ylabel("rms $|\\hat{u}_k|$"); ax.set_ylim(1e-8, 2); ax.legend(frameon=False, fontsize=13, loc="lower left"); panel_label(ax, "e")
    ax = fig.add_subplot(gs[1, 2])
    ax.semilogy(t[1:], np.maximum(np.array(D["onestep"]["mass"])[1:], 1e-9), color=NAVY, lw=2.4, label="one-step, unconstrained")
    ax.semilogy(t[1:], np.maximum(np.array(D["unroll"]["mass"])[1:], 1e-9), color=RED, ls="--", lw=2.2, label="unrolled, unconstrained")
    ax.semilogy(t[1:], np.maximum(np.array(D["cons"]["mass"])[1:], 1e-9), color=GREEN, ls="-.", lw=2.2, label="zero mode removed")
    ax.set_xlabel("time t"); ax.set_ylabel("mean drift $|\\langle u\\rangle(t)-\\langle u\\rangle(0)|$"); ax.legend(frameon=False, fontsize=13); panel_label(ax, "f")
    fig.savefig(os.path.join(HERE, "Fig9.3.5_ks_rollout.png")); plt.close(fig)
