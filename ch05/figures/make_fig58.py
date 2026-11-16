"""Figures 5.8.1 (RPA / ODT / spacing) and 5.8.2 (thin-film DSA snapshots)."""
import sys, json, math, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); sys.path.insert(0, "../ch05/labs")
from bookstyle import *
import numpy as np
from bcp_common import rpa_spsi
LAB = "../ch05/labs/"; FIG = "../ch05/figures/"

rp = json.load(open(LAB + "bcp_rpa.json")); od = json.load(open(LAB + "bcp_odt.json"))
spc = json.load(open(LAB + "bcp_spacing.json"))
fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.6))
ax = axs[0]
sty = {0.0: (GRAY, "o", "-"), 2.0: (BLUE, "s", "--"), 4.0: (RED, "^", None)}
for r in rp["rows"]:
    if r["da"] not in sty: continue
    c, mk, ls = sty[r["da"]]
    uq = np.array(r["uq"]); Sr = np.array(r["Sr"])
    ax.plot(uq, Sr, mk, color=c, ms=8, mfc="none" if r["da"] == 2.0 else c, mew=1.8,
            label=rf"DPD, $\Delta a$ = {r['da']:.0f}")
    if ls:
        qf = np.array(r["qf"]); ff = np.array(r["ff"]); chi = 0.286 * r["da"]
        S = rpa_spsi(ff[:, 0], ff[:, 1], ff[:, 2], chi)
        ax.plot(qf, S, color=c, ls=ls, lw=2, label=rf"RPA, $\chi$ = {chi:.2f}")
        k = np.argmax(S); print(f"RPA Delta a={r['da']}: peak S={S[k]:.2f} at q={qf[k]:.2f}; DPD peak {Sr.max():.2f}")
ax.set_xlabel(r"$q$ ($r_c^{-1}$)"); ax.set_ylabel(r"$S_\psi(q)$ per bead"); ax.set_xlim(0, 3.0)
ax.legend(frameon=False, fontsize=13.5, loc="upper right")
panel_label(ax, "a")

ax = axs[1]
od = np.array(od); chiN = 0.286 * od[:, 0] * 10
ax.errorbar(chiN, od[:, 1], od[:, 2], fmt="s-", color=NAVY, ms=9, lw=2, capsize=3, label="start: lamellae")
ax.errorbar(chiN + 0.3, od[:, 3], od[:, 4], fmt="o--", color=RED, ms=9, lw=2, mfc="none", mew=1.8, capsize=3,
            label="start: disordered")
for xv, txt, c in ((rp["chis"] * 10, "mean-field\nspinodal", GRAY), (rp["fh"], "Fredrickson–\nHelfand", GOLD)):
    ax.axvline(xv, color=c, ls=":", lw=2); ax.text(xv + 0.4, 0.39, txt, fontsize=14, color=GRAY, va="top")
ax.set_xlim(8, 36); ax.set_ylim(0, 0.4)
ax.set_xlabel(r"$\chi N$  ($\chi$ = 0.286 $\Delta a$, $N$ = 10)"); ax.set_ylabel(r"order parameter $\Psi$")
ax.legend(frameon=False, loc="lower right", fontsize=14)
panel_label(ax, "b")

ax = axs[2]
rows = spc["rows"]; Ns = np.array([r["N"] for r in rows]); D = np.array([r["Deq"] for r in rows])
dD = np.array([r["dD"] for r in rows]); a = np.array([r["a"] for r in rows])
ax.errorbar(Ns, D, dD, fmt="o", color=NAVY, ms=10, label="DPD stress-free $D_{eq}$")
Nf = np.linspace(7, 22, 50)
ax.plot(Nf, 1.098 * a.mean() * (0.286 * 15) ** (1 / 6) * Nf ** (2 / 3), color=RED, lw=2,
        label=r"SST: $1.10\,a\,\chi^{1/6}N^{2/3}$")
ax.plot(Nf, D[1] * (Nf / 10) ** 0.5, color=GRAY, ls="--", lw=1.8, label=r"$\propto N^{1/2}$ (coil)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([8, 10, 14, 20]); ax.set_xticklabels(["8", "10", "14", "20"])
ax.set_yticks([5, 6, 8, 10]); ax.set_yticklabels(["5", "6", "8", "10"]); ax.minorticks_off()
ax.set_xlabel(r"chain length $N$"); ax.set_ylabel(r"lamellar period $D$ ($r_c$)")
ax.text(12.5, 5.2, rf"fit: $D\propto N^{{{spc['nu']:.3f}}}$", fontsize=16)
ax.legend(frameon=False, loc="upper left", fontsize=14)
panel_label(ax, "c")
fig.tight_layout(); fig.savefig(FIG + "Fig5.8.1_bcp_theory.png"); plt.close(fig)

fl = np.load(LAB + "bcp_film.npz"); box = fl["box"]; L0 = box[0] / 4
fig, axs = plt.subplots(1, 3, figsize=(17, 6.0))
for ax, case, lab in zip(axs, ("unguided", "dense", "sparse"), "abc"):
    x = fl[case + "_x"]; t = fl[case + "_typ"]
    ax.scatter(x[t == 1, 0], x[t == 1, 1], s=1.2, c=GOLD, alpha=0.5, lw=0)
    ax.scatter(x[t == 0, 0], x[t == 0, 1], s=1.6, c=NAVY, lw=0)
    if case == "dense":
        for m in range(4): ax.plot([m * L0, m * L0], [-1.6, -0.5], color=RED, lw=4, solid_capstyle="butt")
    if case == "sparse":
        for m in range(2): ax.plot([m * 2 * L0, m * 2 * L0], [-1.6, -0.5], color=RED, lw=4, solid_capstyle="butt")
    ax.set_xlim(-0.3, box[0] + 0.3); ax.set_ylim(-1.8, box[1] + 0.3); ax.set_aspect(1)
    ax.set_xlabel(r"$x$ ($r_c$)"); ax.set_ylabel(r"$y$ ($r_c$)")
    h = fl[case + "_hist"][-1]
    panel_label(ax, lab, x=-0.16)
fig.tight_layout(); fig.savefig(FIG + "Fig5.8.2_dsa_film.png"); plt.close(fig)
print("ok")
