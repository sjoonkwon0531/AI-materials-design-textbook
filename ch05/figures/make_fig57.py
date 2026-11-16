"""Figures 5.7.1 (Ising acceleration) and 5.7.2 (DPD basics) from the lab JSON outputs."""
import sys, json, math, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); sys.path.insert(0, "../../ch05/labs")
from bookstyle import *
import numpy as np
from ising_speed import onsager_energy, yang_m, TC

LAB = "../../ch05/labs/"; FIG = "../../ch05/figures/"

# ------------------------------------------------------------------ Fig 5.7.1
sp = json.load(open(LAB + "ising_speed.json")); wf = json.load(open(LAB + "ising_wolff.json"))
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
ax = axs[0, 0]
keys = ["python", "numpy_cb", "numba_random", "numba_cb", "jax_cb", "jax_vmap8"]
labels = ["pure Python", "NumPy checkerboard", "Numba random-site", "Numba checkerboard",
          "JAX checkerboard", "JAX vmap (8 T)"]
cols = [GRAY, BLUE, GOLD, NAVY, RED, PURPLE]
vals = [sp["timing"][k] for k in keys]
y = np.arange(len(keys))[::-1]
ax.barh(y, vals, color=cols, height=0.65)
for yi, v in zip(y, vals):
    ax.text(v * 1.12, yi, f"{v:.0f} ns", va="center", fontsize=15)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xscale("log"); ax.set_xlim(10, 6000)
ax.set_xlabel("time per attempted spin update (ns)")
panel_label(ax, "a", x=-0.45)

ax = axs[0, 1]
Tf = np.linspace(1.4, 3.2, 400)
ax.plot(Tf, [yang_m(t) for t in Tf], color=GRAY, lw=2, label="Yang (exact, L→∞)")
nb = np.array(sp["numba"]); jx = np.array(sp["jax"])
ax.errorbar(nb[:, 0], nb[:, 3], nb[:, 4], fmt="o", color=NAVY, ms=8, label="Numba, L=64")
ax.errorbar(jx[:, 0], jx[:, 3], jx[:, 4], fmt="s", mfc="none", mec=RED, color=RED, ms=10, mew=1.8,
            label="JAX vmap, L=64")
ax.axvline(TC, color=GRAY, ls=":", lw=1.2)
ax.set_xlabel(r"$k_BT/J$"); ax.set_ylabel(r"$\langle |m| \rangle$"); ax.legend(frameon=False, loc="upper right")
panel_label(ax, "b")

ax = axs[1, 0]
ax.plot(Tf, [onsager_energy(t) for t in Tf], color=GRAY, lw=2, label="Onsager (exact, L→∞)")
ax.errorbar(nb[:, 0], nb[:, 1], nb[:, 2], fmt="o", color=NAVY, ms=8, label="Numba, L=64")
ax.errorbar(jx[:, 0], jx[:, 1], jx[:, 2], fmt="s", mfc="none", mec=RED, color=RED, ms=10, mew=1.8,
            label="JAX vmap, L=64")
ax.axvline(TC, color=GRAY, ls=":", lw=1.2)
ax.set_xlabel(r"$k_BT/J$"); ax.set_ylabel(r"energy per spin $e/J$"); ax.legend(frameon=False, loc="upper left")
panel_label(ax, "c")

ax = axs[1, 1]
rows = wf["rows"]; Ls = np.array([r["L"] for r in rows])
tM = np.array([r["tindM"] for r in rows]) * 1e3; tW = np.array([r["tindW"] for r in rows]) * 1e3
ax.loglog(Ls, tM, "o-", color=NAVY, ms=9, lw=2, label="Metropolis (checkerboard)")
ax.loglog(Ls, tW, "s--", color=RED, ms=9, lw=2, mfc="none", mew=1.8, label="Wolff cluster")
ax.set_xlabel("lattice size L"); ax.set_ylabel("wall time per independent sample (ms)")
ax.set_xticks(Ls); ax.set_xticklabels([str(l) for l in Ls]); ax.minorticks_off()
ax.legend(frameon=False, loc="upper left")
ax.text(19, 0.013, f"at $T_c$:  $\\tau_{{\\rm M}}\\propto L^{{{wf['zM']:.2f}}}$,  $\\tau_{{\\rm W}}\\propto L^{{{wf['zW']:.2f}}}$",
        fontsize=16)
panel_label(ax, "d")
fig.tight_layout(); fig.savefig(FIG + "Fig5.7.1_ising_accel.png"); plt.close(fig)

# ------------------------------------------------------------------ Fig 5.7.2
d = json.load(open(LAB + "dpd_basics.json"))
fig, axs = plt.subplots(1, 3, figsize=(17, 5.3))
ax = axs[0]
for lam, c, mk, ls in (("0.5", BLUE, "o", "--"), ("0.65", NAVY, "s", "-")):
    r = np.array(d["p1"][lam])
    ax.errorbar(r[:, 0], r[:, 1], r[:, 2], fmt=mk + ls, color=c, ms=9, lw=2, mfc="none" if lam == "0.5" else c,
                mew=1.8, label=rf"$\lambda$ = {lam}")
ax.axhline(1.0, color=GRAY, ls=":", lw=1.5)
ax.set_xlabel(r"time step $\Delta t$"); ax.set_ylabel(r"measured $k_BT$"); ax.legend(frameon=False, loc="upper left")
panel_label(ax, "a")
ax = axs[1]
xc = np.array(d["xc"]); L = 16.0
styles = {"8.0": (GOLD, ":"), "10.0": (BLUE, "--"), "12.0": (RED, "-."), "15.0": (NAVY, "-")}
for k, (c, ls) in styles.items():
    ax.plot(xc, d["profiles"][k], color=c, ls=ls, lw=2.2, label=rf"$\Delta a$ = {float(k):.0f}")
ax.set_xlabel(r"$x$ (re-centred on A slab)"); ax.set_ylabel(r"A fraction $\phi_A(x)$")
ax.set_ylim(-0.03, 1.08); ax.legend(frameon=False, loc="upper right", fontsize=14)
panel_label(ax, "b")
ax = axs[2]
p3 = np.array(d["p3"])
ax.errorbar(p3[:, 0], p3[:, 3], p3[:, 4], fmt="o", color=NAVY, ms=9, label="DPD slab coexistence")
xx = np.linspace(0, 16, 50)
ax.plot(xx, d["slope0"] * xx, color=RED, lw=2, label=rf"fit: $\chi$ = {d['slope0']:.3f} $\Delta a$")
ax.plot(xx, 0.286 * xx, color=GRAY, ls="--", lw=1.6, label=r"Groot–Warren: 0.286 $\Delta a$")
ax.axhline(2.0, color=GOLD, ls=":", lw=1.8); ax.text(0.5, 2.1, r"critical $\chi_c$ = 2", color=GRAY, fontsize=14)
ax.set_xlim(0, 16); ax.set_ylim(0, 5)
ax.set_xlabel(r"$\Delta a = a_{AB} - a_{AA}$"); ax.set_ylabel(r"Flory–Huggins $\chi$")
ax.legend(frameon=False, loc="upper left", fontsize=14)
panel_label(ax, "c")
fig.tight_layout(); fig.savefig(FIG + "Fig5.7.2_dpd_basics.png"); plt.close(fig)
print("ok")
