"""Figures for 9.5: Fig9.5.1 (DeepXDE heat PINN), Fig9.5.2 (NS rollout snapshots), Fig9.5.3 (errors, spectrum, cost)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
import matplotlib.pyplot as plt
from bookstyle import *

D = "../data/"

def fig1():
    h = np.load(D + "95_dde_heat_results.npz")
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.2), gridspec_kw=dict(wspace=0.38))
    for k, c, ls, lab in (("soft", NAVY, "-", "soft constraints"), ("hard", RED, "--", "hard constraints")):
        ax[0].semilogy(h[f"{k}_steps"], h[f"{k}_metric"], color=c, ls=ls, lw=2.2, marker="o", ms=4, label=lab)
    ax[0].axvline(3000, color=GRAY, ls=":", lw=1.5)
    ax[0].text(2850, 6e-6, "Adam", ha="right", color=GRAY, fontsize=15)
    ax[0].text(3150, 6e-6, "L-BFGS", ha="left", color=GRAY, fontsize=15)
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("relative $L^2$ error (test points)")
    ax[0].set_ylim(4e-6, 1.5); ax[0].legend(frameon=False, loc="upper right")
    xg = np.linspace(0, 1, 101); Xg, Tg = np.meshgrid(xg, xg, indexing="ij")
    err = np.abs(h["soft_pred"] - np.exp(-0.1 * np.pi**2 * Tg) * np.sin(np.pi * Xg))
    im = ax[1].pcolormesh(Tg, Xg, err * 1e4, cmap="magma", shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax[1], pad=0.02); cb.set_label(r"$|u_\theta - u|$  ($10^{-4}$)")
    ax[1].set_xlabel("time $t$"); ax[1].set_ylabel("position $x$")
    cn = h["cn"]
    ax[2].loglog(cn[:, 3] / 1e3, cn[:, 2], color=GREEN, marker="s", ms=8, lw=2, label="Crank–Nicolson")
    for n, e, ms in zip(cn[:, 0], cn[:, 2], cn[:, 3]):
        if int(n) in (25, 100, 800):
            ax[2].annotate(f"$n$={int(n)}", (ms / 1e3, e), textcoords="offset points", xytext=(10, 2), fontsize=14, color=GREEN)
    for k, c, mk, lab in (("soft", NAVY, "o", "PINN, soft"), ("hard", RED, "^", "PINN, hard")):
        s = h[f"{k}_summary"]
        ax[2].loglog(s[:, 1], s[:, 2], color=c, marker=mk, ms=9, lw=1.8, ls="--" if k == "hard" else "-", label=lab)
    ax[2].set_xlabel("CPU time (s)"); ax[2].set_ylabel("relative $L^2$ error")
    ax[2].set_xlim(2e-4, 6e2); ax[2].set_ylim(2e-9, 0.1); ax[2].legend(frameon=False, loc="lower left")
    for a, l in zip(ax, "abc"):
        panel_label(a, l)
    fig.savefig("Fig9.5.1_dde_heat.png"); plt.close(fig)

def fig2():
    r = np.load(D + "95_fno_ns_results.npz")
    truth, pred = r["truth"][0], r["pred"][0]
    times = (1, 5, 10, 20)
    fig, ax = plt.subplots(3, 4, figsize=(15.5, 11.2), gridspec_kw=dict(wspace=0.06, hspace=0.08))
    vmax = np.abs(truth[list(times)]).max()
    for j, tq in enumerate(times):
        im0 = ax[0, j].imshow(truth[tq].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=(0, 1, 0, 1))
        ax[1, j].imshow(pred[tq].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=(0, 1, 0, 1))
        im2 = ax[2, j].imshow((pred[tq] - truth[tq]).T, origin="lower", cmap="PuOr", vmin=-0.1, vmax=0.1, extent=(0, 1, 0, 1))
        ax[2, j].set_xlabel(f"$t$ = {tq}")
        for i in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for i, lab in enumerate(("solver $w$", "FNO rollout $w$", "FNO $-$ solver")):
        ax[i, 0].set_ylabel(lab)
    cb = fig.colorbar(im0, ax=ax[:2, :], fraction=0.02, pad=0.015); cb.set_label("vorticity")
    cb = fig.colorbar(im2, ax=ax[2, :], fraction=0.02, pad=0.015); cb.set_label("difference"); cb.set_ticks([-0.1, 0, 0.1])
    for i, l in enumerate("abc"):
        panel_label(ax[i, 0], l, x=-0.2, y=0.9)
    fig.savefig("Fig9.5.2_ns_rollout.png"); plt.close(fig)

def fig3():
    r = np.load(D + "95_fno_ns_results.npz"); c = np.load(D + "95_compare.npz")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(wspace=0.36))
    t = np.arange(1, 21)
    ax[0].semilogy(t, r["e_pers1"].mean(0), color=GRAY, ls=":", marker="s", ms=5, lw=2, label="persistence (1 step)")
    ax[0].semilogy(t, r["e_roll"].mean(0), color=RED, ls="-", marker="o", ms=5, lw=2.2, label="FNO rollout from $w_0$")
    ax[0].semilogy(t, r["e_one"].mean(0), color=NAVY, ls="--", marker="^", ms=5, lw=2.2, label="FNO one step")
    ax[0].fill_between(t, np.percentile(r["e_roll"], 5, 0), np.percentile(r["e_roll"], 95, 0), color=RED, alpha=0.15, lw=0)
    ax[0].set_xlabel("time $t$"); ax[0].set_ylabel("relative $L^2$ error"); ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[0].set_ylim(8e-4, 1.5); ax[0].legend(frameon=False, loc="upper right", fontsize=14)
    k = np.arange(33)
    ax[1].loglog(k[1:], r["spec_true"][1:], color=NAVY, lw=2.4, marker="o", ms=4, label="solver")
    ax[1].loglog(k[1:], r["spec_pred"][1:], color=RED, lw=2.0, ls="--", marker="^", ms=4, label="FNO rollout")
    ax[1].loglog(k[1:], r["spec_err"][1:], color=GOLD, lw=2.0, ls="-.", label="FNO $-$ solver")
    ax[1].axvline(12, color=GRAY, ls=":", lw=1.5); ax[1].text(12.8, 3e-2, "FNO modes", color=GRAY, fontsize=14)
    ax[1].set_xlabel("wavenumber $|k|$"); ax[1].set_ylabel("enstrophy spectrum at $t$ = 20")
    ax[1].set_ylim(1e-14, 5); ax[1].legend(frameon=False, loc="lower left", fontsize=14)
    rows = c["rows"]
    ax[2].loglog(rows[:, 5], rows[:, 4], color=GREEN, marker="s", ms=9, lw=2, label="pseudo-spectral solver")
    for N, e, ms in zip(rows[:, 0], rows[:, 4], rows[:, 5]):
        off = {64: (9, -4), 48: (-34, 8), 12: (-36, 2), 16: (-36, -6)}.get(int(N), (8, 6))
        ax[2].annotate(f"{int(N)}$^2$", (ms, e), textcoords="offset points", xytext=off, fontsize=14, color=GREEN)
    ax[2].loglog([c["ms_fno"]], [c["e_fno"][19]], color=RED, marker="o", ms=13, ls="none", label="FNO (inference only)")
    ax[2].loglog([1e3 * c["pinn_cpu_hard"] / 5], [c["e_pinn_hard"][4]], color=PURPLE, marker="^", ms=13, ls="none", label="PINN ($t$ = 5)")
    ax[2].set_xlabel("CPU time per unit time (ms)"); ax[2].set_xlim(0.2, 1e5); ax[2].set_ylabel("relative $L^2$ error at $t$ = 20")
    ax[2].legend(frameon=False, loc="center right", bbox_to_anchor=(1.02, 0.52), fontsize=14)
    for a, l in zip(ax, "abc"):
        panel_label(a, l)
    fig.savefig("Fig9.5.3_ns_compare.png"); plt.close(fig)

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3}[w]()
