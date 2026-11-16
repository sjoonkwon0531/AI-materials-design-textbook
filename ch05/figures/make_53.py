"""Figures for Sec. 5.3 (KMC): Fig5.3.1 SSA vs ODE, Fig5.3.2 adatom Arrhenius, Fig5.3.3 ZGB."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
from matplotlib.colors import ListedColormap
LAB = "../../ch05/labs/"
OUT = "../../ch05/figures/"
KB = 8.617333262e-5

# ---------------- Fig 5.3.1
d = np.load(LAB + "53_ssa_fig.npz")
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.25, 0.9, 1.25]})
t = d["tgrid"]
for k, tr in enumerate(d["tr100"]):
    ax[0].step(t, tr, where="post", color=GOLD, lw=1.0, alpha=0.9, label="SSA, N = 100" if k == 0 else None)
for k, tr in enumerate(d["tr1000"]):
    ax[0].step(t, tr, where="post", color=BLUE, lw=1.0, alpha=0.9, label="SSA, N = 1000" if k == 0 else None)
ax[0].plot(t, d["exact"], color=NAVY, lw=2.6, ls="--", label="ODE  exp(At)X$_0$")
ax[0].set_xlabel("time"); ax[0].set_ylabel("$X_3/N$")
ax[0].set_ylim(0, 0.72); ax[0].legend(loc="upper right", frameon=False, ncol=1)
panel_label(ax[0], "a")
sN = d["stdN"]
Ns = np.logspace(1.8, 4.2, 50)
ax[1].loglog(Ns, np.sqrt(2 / 9 / Ns), color=GRAY, lw=2, ls="--", label="binomial theory")
ax[1].loglog(sN[:, 0], sN[:, 2], "o", color=RED, ms=10, label="SSA, t = 8")
ax[1].set_xlabel("molecule number N"); ax[1].set_ylabel("std of $X_3/N$")
ax[1].legend(frameon=False, loc="lower left")
panel_label(ax[1], "b", x=-0.24)
rec = d["rec"]; tt = np.arange(rec.size) * float(d["dtr"])
m = tt <= 1000
ax[2].plot(tt[m], rec[m], color=PURPLE, lw=0.6)
for r, ls in zip(d["roots"], ["-", ":", "-"]):
    ax[2].axhline(r, color=GRAY, lw=1.6, ls=ls)
ax[2].set_xlabel("time"); ax[2].set_ylabel("X (Schlögl)")
ax[2].text(1010, d["roots"][0], "stable", va="center", fontsize=14, color=GRAY)
ax[2].text(1010, d["roots"][1], "unstable", va="center", fontsize=14, color=GRAY)
ax[2].text(1010, d["roots"][2], "stable", va="center", fontsize=14, color=GRAY)
ax[2].set_xlim(0, 1000)
panel_label(ax[2], "c")
fig.tight_layout(w_pad=2.0)
fig.savefig(OUT + "Fig5.3.1_ssa_vs_ode.png"); plt.close(fig)

# ---------------- Fig 5.3.2
d = np.load(LAB + "53_lattice_fig.npz")
Ts = d["Ts"]; x = 1000.0 / Ts
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
Dex = d["nu"] * np.exp(-d["Ea"] / (KB * Ts))
xx = np.linspace(x.min() * 0.97, x.max() * 1.02, 50)
ax[0].semilogy(xx, d["nu"] * np.exp(-d["Ea"] / (KB * 1000.0 / xx)), color=GRAY, lw=2, ls="--", label="exact, single adatom")
ax[0].semilogy(x, d["Dsingle"], "o", color=NAVY, ms=10, label="KMC, single adatom")
ax[0].semilogy(x, d["D0"], "s", color=BLUE, ms=9, mfc="none", mew=2, label=r"KMC, $\theta$ = 0.1, $E_b$ = 0")
ax[0].semilogy(x, d["D1"], "^", color=RED, ms=10, label=r"KMC, $\theta$ = 0.1, $E_b$ = 0.1 eV")
p = np.polyfit(1 / (KB * Ts), np.log(d["D1"]), 1)
ax[0].semilogy(xx, np.exp(np.polyval(p, 1 / (KB * 1000.0 / xx))), color=RED, lw=1.5, ls=":")
ax[0].set_xlabel("1000 / T  (K$^{-1}$)"); ax[0].set_ylabel("D  (a$^2$ s$^{-1}$)")
ax[0].legend(frameon=False, loc="lower left", fontsize=13.5)
panel_label(ax[0], "a")
ax[1].plot(Ts, d["D0"] / Dex, "s-", color=BLUE, ms=9, mfc="none", mew=2, label=r"$E_b$ = 0")
ax[1].plot(Ts, d["D1"] / Dex, "^-", color=RED, ms=10, label=r"$E_b$ = 0.1 eV")
ax[1].set_xlabel("T  (K)"); ax[1].set_ylabel(r"$D/D_{\mathrm{single}}$")
ax[1].set_yscale("log"); ax[1].legend(frameon=False, loc="lower right")
panel_label(ax[1], "b")
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT + "Fig5.3.2_adatom_arrhenius.png"); plt.close(fig)

# ---------------- Fig 5.3.3
z = d["zgb"]
fig = plt.figure(figsize=(15.5, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
a0 = fig.add_subplot(gs[0])
a0.plot(z[:, 0], z[:, 1], "o-", color=RED, ms=7, label=r"$\theta_{\mathrm{CO}}$")
a0.plot(z[:, 0], z[:, 2], "s--", color=BLUE, ms=7, label=r"$\theta_{\mathrm{O}}$")
a0.plot(z[:, 0], z[:, 3] * 3, "^-", color=NAVY, ms=8, lw=2.2, label=r"3 $\times$ $R_{\mathrm{CO_2}}$")
for yv in (0.3874, 0.5256):
    a0.axvline(yv, color=GRAY, ls=":", lw=1.5)
a0.text(0.3844, 0.80, "$y_1$", ha="right", fontsize=18, color=GRAY)
a0.text(0.5296, 0.60, "$y_2$", ha="left", fontsize=18, color=GRAY)
a0.set_xlabel("CO fraction in gas, y"); a0.set_ylabel("coverage / rate")
a0.set_ylim(-0.03, 1.1)
a0.legend(frameon=False, loc="center left", bbox_to_anchor=(0.22, 0.55))
panel_label(a0, "a")
cmap = ListedColormap(["white", RED, BLUE])
for k, (key, lab) in enumerate([("snap1", "b"), ("snap2", "c")]):
    a = fig.add_subplot(gs[k + 1])
    a.imshow(d[key], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True)
    panel_label(a, lab, x=-0.08)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT + "Fig5.3.3_zgb.png"); plt.close(fig)
s2 = d["snap2"]
print("snap1 coverages CO,O:", (d["snap1"] == 1).mean(), (d["snap1"] == 2).mean())
print("snap2 coverages CO,O:", (s2 == 1).mean(), (s2 == 2).mean())
