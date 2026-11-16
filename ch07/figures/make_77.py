"""Figures for Sec. 7.7 from ../labs/77_results.json and ../labs/77_md.npz. Prints every plotted number."""
import os, sys, json, logging
import numpy as np
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = os.path.join(HERE, "..", "labs")
res = json.load(open(os.path.join(LABS, "77_results.json")))
NS = [100, 300, 1000]
STY = {"mlp": ("MLP (xyz)", GRAY, "x", ":"), "mlp_aug": ("MLP (xyz) + rot. aug.", GOLD, "s", "--"),
       "mlp_frame": ("MLP (molecular frame)", PURPLE, "D", "-."),
       "schnet": ("SchNet-like (invariant)", BLUE, "o", "-"), "nequip": ("NequIP-like (equivariant)", RED, "^", "-")}

# ---------- Fig 7.7.1 learning curves ----------
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(wspace=0.3))
for k, (lab, c, m, ls) in STY.items():
    F = [1e3 * res[f"{k}_N{n}"]["300K"]["F_mae"] for n in NS]
    E = [1e3 * res[f"{k}_N{n}"]["300K"]["E_mae"] for n in NS]
    sF = np.polyfit(np.log(NS), np.log(F), 1)[0]; sE = np.polyfit(np.log(NS), np.log(E), 1)[0]
    print("Fig7.7.1 %-26s F %s meV/A slope %.2f | E %s meV slope %.2f" % (lab, np.round(F, 1), sF, np.round(E, 1), sE))
    axs[0].plot(NS, F, ls=ls, marker=m, color=c, lw=2, ms=9, label=lab)
    axs[1].plot(NS, E, ls=ls, marker=m, color=c, lw=2, ms=9, label=lab)
for ax, yl in zip(axs, ["force MAE (meV/Å)", "energy MAE (meV)"]):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training frames N"); ax.set_ylabel(yl)
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
axs[0].axhline(1e3 * 0.715, color=GRAY, lw=1, ls=(0, (1, 3)))
axs[0].text(110, 1e3 * 0.715 * 1.08, "predict F = 0", fontsize=13, color=GRAY)
for k in ["schnet", "nequip"]:
    if f"{k}_N1000_long" in res:
        lab, c, m, ls = STY[k]; v = 1e3 * res[f"{k}_N1000_long"]["300K"]["F_mae"]
        print("Fig7.7.1 %-26s N=1000, 6000 steps: F %.1f meV/A" % (lab, v))
        axs[0].plot([1000], [v], marker=m, ms=11, mfc="white", mec=c, mew=2, ls="none",
                    label="3x steps" if k == "nequip" else None)
axs[0].legend(*axs[1].get_legend_handles_labels(), frameon=False, fontsize=13, loc="center right",
              bbox_to_anchor=(1.0, 0.55))
panel_label(axs[0], "a"); panel_label(axs[1], "b")
fig.savefig(os.path.join(HERE, "Fig7.7.1_learning_curves.png")); plt.close(fig)

# ---------- Fig 7.7.2 extrapolation and rotation ----------
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(wspace=0.3))
Ts = ["300K", "600K", "1000K"]
ax = axs[0]
for k in ["mlp_frame", "schnet", "nequip"]:
    lab, c, m, ls = STY[k]
    v = [1e3 * res[f"{k}_N1000"][t]["F_mae"] for t in Ts]
    print("Fig7.7.2a %-26s F MAE 300/600/1000 K: %s  ratio 1000/300 %.2f" % (lab, np.round(v, 1), v[2] / v[0]))
    ax.plot([300, 600, 1000], v, ls=ls, marker=m, color=c, lw=2, ms=9, label=lab)
ax.set_xlabel("test temperature (K)"); ax.set_ylabel("force MAE (meV/Å)"); ax.set_xticks([300, 600, 1000])
ax.set_yscale("log"); ax.legend(frameon=False, fontsize=13, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax.axvspan(250, 350, color=GRAY, alpha=0.12)
panel_label(ax, "a")
ax = axs[1]
keys = list(STY)
x = np.arange(len(keys)); w = 0.38
a = [1e3 * res[f"{k}_N1000"]["300K"]["F_mae"] for k in keys]
b = [1e3 * res[f"{k}_N1000"]["300K_rot"]["F_mae"] for k in keys]
re = [res[f"{k}_N1000"]["rot_err_F"] for k in keys]
print("Fig7.7.2b F MAE orig:", np.round(a, 1), "rotated:", np.round(b, 1), "rot err:", ["%.1e" % r for r in re])
ax.bar(x - w / 2, a, w, color=NAVY, label="test frames as sampled")
ax.bar(x + w / 2, b, w, color=GOLD, hatch="//", edgecolor="white", label="same frames, randomly rotated")
for i, r in enumerate(re):
    ax.text(i, max(a[i], b[i]) * 1.04, "%.0e" % r if r < 1e-2 else "%.2f" % r, ha="center", fontsize=12, color=RED)
ax.set_xticks(x); ax.set_xticklabels(["MLP\nxyz", "MLP xyz\n+aug", "MLP\nframe", "SchNet-\nlike", "NequIP-\nlike"], fontsize=14)
ax.set_ylabel("force MAE (meV/Å), N = 1000"); ax.set_ylim(0, max(max(a), max(b)) * 1.25)
ax.legend(frameon=False, fontsize=13, loc="upper right")
panel_label(ax, "b")
fig.savefig(os.path.join(HERE, "Fig7.7.2_extrapolation_rotation.png")); plt.close(fig)

# ---------- Fig 7.7.3 MD ----------
p = os.path.join(LABS, "77_md.npz")
if os.path.exists(p):
    md = np.load(p, allow_pickle=True); keys = list(md["keys"]); status = list(md["status"])
    MSTY = {"xTB": ("GFN2-xTB", "black", "-"), "xTB_seed1": ("GFN2-xTB, other velocities", GRAY, "-."), "mlp_N1000": ("MLP (xyz)", GRAY, ":"),
            "mlp_frame_N1000": ("MLP (frame)", PURPLE, "-."), "schnet_N1000": ("SchNet-like", BLUE, "-"),
            "nequip_N1000": ("NequIP-like", RED, "--"), "mlp_aug_N1000": ("MLP + aug", GOLD, "--"),
            "schnet_N100": ("SchNet-like N=100", BLUE, ":"), "nequip_N100": ("NequIP-like N=100", RED, ":")}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw=dict(wspace=0.32))
    for k in ["xTB", "nequip_N1000", "schnet_N1000", "mlp_frame_N1000", "mlp_N1000", "mlp_aug_N1000"]:
        lab, c, ls = MSTY[k]
        t = md[f"{k}__t"]; E = md[f"{k}__Etot"]
        if len(E) == 0: continue
        axs[0].plot(t, 1e3 * (E - E[0]), color=c, ls=ls, lw=1.6, label=lab)
        axs[2].plot(t, np.maximum(md[f"{k}__L"], 1e-17), color=c, ls=ls, lw=1.6, marker="o" if len(t) < 20 else None, ms=5)
    axs[0].set_xlabel("time (ps)"); axs[0].set_ylabel(r"$E_{\mathrm{tot}}(t)-E_{\mathrm{tot}}(0)$ (meV)")
    axs[0].set_ylim(-6, 3)
    axs[2].set_xlabel("time (ps)"); axs[2].set_ylabel(r"$|\mathbf{L}|$ (ASE units)"); axs[2].set_yscale("log")
    axs[2].set_ylim(1e-17, 3)
    ax = axs[1]; bins = np.linspace(0.90, 1.04, 36); ax.set_xlim(0.90, 1.075); ax.set_ylim(0, 36)
    for k in ["xTB", "xTB_seed1", "nequip_N1000", "schnet_N1000"]:
        lab, c, ls = MSTY[k]; bo = md[f"{k}__bonds"]
        if bo.size == 0: continue
        ax.hist(bo[0], bins=bins, density=True, histtype="step", color=c, ls=ls, lw=2,
                label=lab if k == "xTB_seed1" else None)
    ax.set_xlabel("O–H bond length (Å)"); ax.set_ylabel("probability density")
    for a_, l_ in zip(axs, "abc"): panel_label(a_, l_)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, -0.1))
    fig.savefig(os.path.join(HERE, "Fig7.7.3_nve_md.png")); plt.close(fig)
    print("Fig7.7.3 statuses:", dict(zip(keys, status)))
    for k in keys:
        L = md[f"{k}__L"]; print("Fig7.7.3 %-18s frames %4d  max|L| %.2e  final|L| %.2e" % (k, len(L), L.max(), L[-1]))
print("done")
