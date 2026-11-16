"""Figures for section 5.2: Fig5.2.1 (LJ MC), Fig5.2.2 (autocorrelation), Fig5.2.3 (FSS)."""
import sys, logging
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from bookstyle import *
import numpy as np

LAB = "../../ch05/labs/"
OUT = "../../ch05/figures/"

# ------------------------------------------------------------------ Fig5.2.1
r = np.load(LAB + "52_ljmc_results.npz")
e = np.load(LAB + "52_ljmc_eos.npz")
fig, axs = plt.subplots(2, 2, figsize=(12, 9.4))
ax = axs[0, 0]
ax.plot(e["rho"], e["u"], color=NAVY, lw=2.2, label="EOS (Thol et al. 2016)")
ax.errorbar(r["rhoA"], r["uA"], yerr=r["uA_err"], fmt="o", ms=9, color=RED, mfc="white", mew=2,
            label="NVT MC, N = 256")
ax.set_xlabel(r"density $\rho^{*}$"); ax.set_ylabel(r"$U/N$  ($\varepsilon$)")
ax.legend(frameon=False, loc="lower left"); panel_label(ax, "a")
ax = axs[0, 1]
ax.plot(e["rho"], e["p"], color=NAVY, lw=2.2, label="EOS")
ax.errorbar(r["rhoA"], r["pA"], yerr=r["pA_err"], fmt="o", ms=9, color=RED, mfc="white", mew=2, label="NVT MC")
ax.set_xlabel(r"density $\rho^{*}$"); ax.set_ylabel(r"pressure $P^{*}$")
ax.legend(frameon=False, loc="upper left"); panel_label(ax, "b")
ax = axs[1, 0]
sw = np.arange(len(r["rhoB_series1"]))
ax.plot(sw, r["rhoB_series3"], color=BLUE, lw=0.6, alpha=0.8, label=r"$P^{*}=3.0$")
ax.plot(sw, r["rhoB_series1"], color=GREEN, lw=0.6, alpha=0.8, label=r"$P^{*}=1.0$")
ax.plot(sw, r["rhoB_wrong"], color=RED, lw=0.6, alpha=0.8, label=r"$P^{*}=1.0$, no $(N+1)\ln V$ term")
for v, c in zip(r["rhoB_eos"], [GREEN, BLUE]):
    ax.axhline(v, color="black", ls="--", lw=1.6)
ax.set_xlabel("production sweep"); ax.set_ylabel(r"density $\rho^{*}$")
ax.set_ylim(0.40, 0.96)
leg = ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=15, handlelength=1.4)
for h in leg.get_lines(): h.set_linewidth(3)
panel_label(ax, "c")
ax = axs[1, 1]
ax.plot(r["zc"], r["prof"], color=NAVY, lw=2.0, label="hard walls (z)")
ax.axhline(0.8, color=GRAY, ls="--", lw=1.6, label="periodic: uniform 0.8")
ax.set_xlabel(r"$z\,/\,\sigma$"); ax.set_ylabel(r"local density $\rho(z)$")
ax.set_xlim(0, float(r["Lzw"])); ax.legend(frameon=False, loc="upper center")
ax.set_ylim(0, 2.6)
panel_label(ax, "d")
fig.tight_layout()
fig.savefig(OUT + "Fig5.2.1_lj_mc.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig5.2.2
d = np.load(LAB + "52_tau_results.npz")
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4))
ax = axs[0]
tm = np.arange(len(d["rho_m32"]))
ax.semilogy(tm, np.clip(d["rho_m32"], 1e-4, None), color=RED, lw=2.0, label="Metropolis")
rw = d["rho_w32"]
kcut = int(np.argmax(rw < 0.0))            # plot the Wolff ACF only until it first hits noise (< 0)
tw = np.arange(kcut) * float(d["f32"])
ax.semilogy(tw, rw[:kcut], color=NAVY, lw=2.0, ls="--", label="Wolff")
ax.set_xlim(0, 800); ax.set_ylim(1e-3, 1.2)
ax.set_xlabel("time (sweeps)"); ax.set_ylabel(r"autocorrelation $\rho_{m^2}(t)$")
ax.legend(frameon=False); panel_label(ax, "a")
ins = ax.inset_axes([0.20, 0.16, 0.36, 0.36])
ins.semilogy(np.arange(16), np.clip(d["rho_m32"][:16], 1e-3, None), color=RED, lw=2)
ins.semilogy(tw, rw[:kcut], color=NAVY, lw=2, ls="--", marker="s", ms=4)
ins.set_xlim(0, 12); ins.set_ylim(1e-2, 1.1); ins.tick_params(labelsize=12)
ins.text(6, 0.5, "first 12 sweeps", fontsize=13, ha="center")
ax = axs[1]
ax.errorbar(d["Lm"], d["tm"], yerr=d["tm_err"], fmt="o", ms=9, color=RED, mfc="white", mew=2, label="Metropolis")
ax.errorbar(d["Lw"], d["tw"], yerr=d["tw_err"], fmt="s", ms=9, color=NAVY, mfc="white", mew=2, label="Wolff")
LL = np.array([6, 160])
cm = np.exp(np.mean(np.log(d["tm"]) - float(d["zm"]) * np.log(d["Lm"])))
cw = np.exp(np.mean(np.log(d["tw"]) - float(d["zw_m"]) * np.log(d["Lw"])))
ax.loglog(LL, cm * LL ** float(d["zm"]), color=RED, lw=1.4, ls=":")
ax.loglog(LL, cw * LL ** float(d["zw_m"]), color=NAVY, lw=1.4, ls=":")
ax.text(9, 300, rf"$z \approx {float(d['zm']):.2f}$", color=RED, fontsize=18)
ax.text(30, 3.2, rf"$z \approx {float(d['zw_m']):.2f}$", color=NAVY, fontsize=18)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"lattice size $L$"); ax.set_ylabel(r"$\tau_{\mathrm{int}}(m^2)$  (sweeps)")
ax.legend(frameon=False, loc="upper left"); panel_label(ax, "b")
ax = axs[2]
blk = d["blk"]
lv = np.arange(len(blk))
ax.plot(lv, blk / blk[0], "o-", color=NAVY, ms=8, lw=2)
ax.axhline(np.sqrt(2 * d["tm"][2]), color=RED, ls="--", lw=1.8, label=r"$\sqrt{2\tau_{\mathrm{int}}}$")
ax.set_xlabel(r"blocking level $k$  (block = $2^k$ sweeps)")
ax.set_ylabel("error / naive error")
ax.legend(frameon=False, loc="lower right"); panel_label(ax, "c")
fig.tight_layout()
fig.savefig(OUT + "Fig5.2.2_autocorr.png")
plt.close(fig)

# ------------------------------------------------------------------ Fig5.2.3
f2 = np.load(LAB + "52_fss2d_results.npz")
f3 = np.load(LAB + "52_fss3d_results.npz")
fig, axs = plt.subplots(2, 2, figsize=(12, 9.6))
styles = [(NAVY, "-"), (RED, "--"), (BLUE, "-."), (GREEN, ":"), (PURPLE, "-"), (GOLD, "--")]
ax = axs[0, 0]
for (L, (c, ls)) in zip([8, 16, 32, 64], styles):
    ax.plot(f2["T"], f2[f"U_{L}"], color=c, ls=ls, lw=2.2, label=f"L = {L}")
ax.axvline(float(f2["Tc"]), color=GRAY, lw=1.2)
ax.set_xlim(2.15, 2.40); ax.set_ylim(0.35, 0.68)
ax.set_xlabel(r"$T$"); ax.set_ylabel(r"Binder cumulant $U_4$")
ax.legend(frameon=False, loc="lower left"); panel_label(ax, "a")
ax = axs[0, 1]
mk = ["o", "s", "^", "D"]
for (L, (c, ls), m) in zip([8, 16, 32, 64], styles, mk):
    x, y = f2[f"cx_{L}"], f2[f"cy_{L}"]
    sel = (x > -3) & (x < 3)
    ax.plot(x[sel][::3], y[sel][::3], m, ms=6, color=c, mfc="none", mew=1.4, label=f"L = {L}")
ax.set_xlim(-2, 2.2)
ax.set_xlabel(r"$t\,L^{1/\nu}$,  $t=(T-T_c)/T_c$"); ax.set_ylabel(r"$\chi'\,L^{-\gamma/\nu}$")
ax.legend(frameon=False, loc="upper left"); panel_label(ax, "b")
ax = axs[1, 0]
for (L, (c, ls)) in zip([6, 8, 12, 16, 24, 32], styles):
    ax.plot(f3["T"], f3[f"U_{L}"], color=c, ls=ls, lw=2.2, label=f"L = {L}")
ax.axvline(1 / 0.221654626, color=GRAY, lw=1.2)
ax.set_xlim(4.47, 4.55); ax.set_ylim(0.36, 0.60)
ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$U_4$  (3D)")
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=14); panel_label(ax, "c")
ax = axs[1, 1]
w = float(f3["w_exp"])
xx = f3["La"] ** (-w)
ax.errorbar(xx, f3["Ts"], yerr=f3["eT"], fmt="o", ms=9, color=NAVY, mfc="white", mew=2, label=r"crossing $T^{*}(L,2L)$")
xs = np.linspace(0, xx.max() * 1.05, 50)
ax.plot(xs, f3["coef"][0] + f3["coef"][1] * xs, color=RED, lw=1.8, ls="--", label="fit")
ax.axhline(1 / 0.221654626, color=GRAY, lw=1.2, ls=":", label=r"$T_c$ = 4.5115 (ref.)")
offs = {6: (10, 6), 8: (10, 6), 12: (10, -26), 16: (-40, -34)}
for L, x0, y0 in zip(f3["La"], xx, f3["Ts"]):
    ax.annotate(f"L={int(L)}", (x0, y0), textcoords="offset points", xytext=offs[int(L)], fontsize=14)
ax.set_xlabel(r"$L^{-(\omega+1/\nu)}$"); ax.set_ylabel(r"$T^{*}$")
ax.set_ylim(4.500, 4.5135)
ax.legend(frameon=False, loc="lower left"); panel_label(ax, "d")
fig.tight_layout()
fig.savefig(OUT + "Fig5.2.3_fss.png")
plt.close(fig)
print("saved")
