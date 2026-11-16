"""Figures for 9.6: Fig9.6.1 (CH snapshots: solver vs FNO rollouts), Fig9.6.2 (error, mass, free energy, growth law),
Fig9.6.3 (out-of-distribution mean composition)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
import matplotlib.pyplot as plt
from bookstyle import *

r = np.load("../data/96_fno_ch_results.npz")
t = r["t"]; t1 = t[1:]
STY = {"free": (BLUE, ":", "s", "FNO"), "cons": (GOLD, "-.", "^", "FNO + zero mode"),
       "cons_pf": (RED, "--", "o", "FNO + zero mode + pushforward")}

def fig1():
    rows = [("snap_solver", "solver"), ("snap_cons", "FNO + zero mode"), ("snap_cons_pf", "+ pushforward")]
    fig, ax = plt.subplots(3, 4, figsize=(14.5, 11.2), gridspec_kw=dict(wspace=0.05, hspace=0.07))
    for i, (key, lab) in enumerate(rows):
        for j, tq in enumerate((5, 12.5, 25, 50)):
            im = ax[i, j].imshow(r[key][0, j].T, origin="lower", cmap="RdBu_r", vmin=-1.1, vmax=1.1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 2:
                ax[i, j].set_xlabel(f"$t$ = {tq:g}")
        ax[i, 0].set_ylabel(lab)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015); cb.set_label("composition $c$")
    for i, l in enumerate("abc"):
        panel_label(ax[i, 0], l, x=-0.2, y=0.9)
    fig.savefig("Fig9.6.1_ch_rollout.png"); plt.close(fig)

def fig2():
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.2), gridspec_kw=dict(wspace=0.3, hspace=0.3))
    a = ax[0, 0]
    a.semilogy(t1[1:], r["frozen"].mean(0), color=GRAY, ls="-", lw=1.8, label="frozen at $c$(2.5)")
    short = {"free": "FNO rollout", "cons": "+ zero mode", "cons_pf": "+ pushforward"}
    for tag, (c, ls, mk, lab) in STY.items():
        a.semilogy(t1[1:], r[f"er_{tag}"].mean(0), color=c, ls=ls, marker=mk, ms=5, lw=2.2, label=short[tag])
    a.semilogy(t[2:], r["e1_cons_pf"][:, 1:].mean(0), color=NAVY, ls="-", marker="D", ms=4, lw=1.8, label="one step")
    a.set_xlabel("time $t$"); a.set_ylabel("relative $L^2$ error"); a.set_ylim(1.5e-3, 3)
    a.legend(frameon=False, fontsize=13, loc="lower center", ncol=2, columnspacing=1.2)
    a = ax[0, 1]
    for tag, (c, ls, mk, lab) in STY.items():
        dr = np.maximum(r[f"drift_{tag}"].max(0)[1:], 1e-18)
        a.semilogy(t1[1:], dr, color=c, ls=ls, marker=mk, ms=5, lw=2.2, label=lab)
    a.axhline(2.2e-16, color=GRAY, ls=":", lw=1.5); a.text(26, 5e-16, "machine epsilon", color=GRAY, fontsize=14)
    a.set_xlabel("time $t$"); a.set_ylabel("max mass drift $|\\langle c\\rangle(t)-\\langle c\\rangle(2.5)|$")
    a.set_ylim(1e-18, 1); a.legend(frameon=False, fontsize=12.5, loc="center right")
    a = ax[1, 0]
    a.plot(t1, r["F_solver"].mean(0), color=NAVY, lw=2.6, label="solver")
    for tag, (c, ls, mk, lab) in STY.items():
        a.plot(t1, r[f"F_{tag}"].mean(0), color=c, ls=ls, marker=mk, ms=5, lw=2.2, label=lab)
    a.set_xlabel("time $t$"); a.set_ylabel("free energy $F$ (mean of 40)"); a.set_ylim(1.2, 5.5); a.legend(frameon=False, fontsize=12.5)
    a = ax[1, 1]
    a.loglog(t1, r["L_solver"].mean(0), color=NAVY, lw=2.6, label="solver")
    for tag, (c, ls, mk, lab) in STY.items():
        a.loglog(t1, r[f"L_{tag}"].mean(0), color=c, ls=ls, marker=mk, ms=5, lw=2.2, label=lab)
    a.loglog(t1[1:], 1.25 * t1[1:] ** (1 / 3), color=GREEN, ls=(0, (1, 1)), lw=2, label="$t^{1/3}$")
    a.set_xlabel("time $t$"); a.set_ylabel("domain size $L(t)$"); a.set_xticks([2.5, 5, 10, 25, 50]); a.set_xticklabels(["2.5", "5", "10", "25", "50"])
    a.set_yticks([1.5, 2, 3, 4]); a.set_yticklabels(["1.5", "2", "3", "4"]); a.minorticks_off(); a.legend(frameon=False, fontsize=12.5, loc="upper left")
    for a, l in zip(ax.ravel(), "abcd"):
        panel_label(a, l)
    fig.savefig("Fig9.6.2_ch_structure.png"); plt.close(fig)

def fig3():
    fig = plt.figure(figsize=(17.5, 7.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.12, 1.75], wspace=0.08, hspace=0.08)
    snap = r["ood_snap_0.5"]                                # (2, 3, 64, 64): solver / FNO at t = 2.5, 5, 50
    for i, lab in enumerate(("solver", "FNO rollout")):
        for j, tq in enumerate((2.5, 5, 50)):
            a = fig.add_subplot(gs[i, j])
            im = a.imshow(snap[i, j].T, origin="lower", cmap="RdBu_r", vmin=-1.1, vmax=1.1)
            a.set_xticks([]); a.set_yticks([])
            if i == 1:
                a.set_xlabel(f"$t$ = {tq:g}")
            if j == 0:
                a.set_ylabel(lab); panel_label(a, "ab"[i], x=-0.22, y=0.88)
    a = fig.add_subplot(gs[:, 4])
    d = np.load("../data/96_ch_64.npz"); am = np.abs(d["m_test"])
    xs = [0.05, 0.15, 0.3, 0.4, 0.5]
    for tag, (c, ls, mk, lab) in STY.items():
        e = r[f"er_{tag}"][:, 8]
        ys = [e[am < 0.1].mean(), e[am >= 0.1].mean()] + [r[f"ood_er_{tag}_{m}"][:, 8].mean() for m in (0.3, 0.4, 0.5)]
        a.plot(xs, ys, color=c, ls=ls, marker=mk, ms=8, lw=2.2, label=lab)
    a.axvspan(0, 0.2, color=ICE, alpha=0.5, lw=0); a.text(0.1, 0.06, "training range", ha="center", fontsize=15, color=NAVY)
    a.set_xlabel("mean composition $|m|$"); a.set_ylabel("rollout error at $t$ = 25"); a.set_xlim(0, 0.55); a.set_ylim(0, None)
    a.legend(frameon=False, fontsize=13.5, loc="upper left"); panel_label(a, "c", x=-0.16)
    fig.savefig("Fig9.6.3_ch_ood.png"); plt.close(fig)

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3"]):
        {"1": fig1, "2": fig2, "3": fig3}[w]()
