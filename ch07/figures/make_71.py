"""Figures for sec 7.1 (run from ch7/figs). Reads lab npz files in ../labs."""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import matplotlib.pyplot as plt
from scipy.special import erf
L = "../labs/"
which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
STY = {"ReLU": (RED, "-", None), "GELU": (BLUE, "--", None), "SiLU": (NAVY, "-", None), "tanh": (GREEN, ":", None),
       "sigmoid": (GOLD, "-.", None)}

if "1" in which:   # activation functions and derivatives
    z = np.linspace(-4, 4, 2001)
    sg = 1 / (1 + np.exp(-z)); Phi = 0.5 * (1 + erf(z / np.sqrt(2))); phi = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    f = {"sigmoid": sg, "tanh": np.tanh(z), "ReLU": np.maximum(z, 0), "GELU": z * Phi, "SiLU": z * sg}
    d1 = {"sigmoid": sg * (1 - sg), "tanh": 1 - np.tanh(z) ** 2, "ReLU": (z > 0).astype(float), "GELU": Phi + z * phi,
          "SiLU": sg * (1 + z * (1 - sg))}
    d2 = {k: np.gradient(v, z) for k, v in d1.items()}
    d2["ReLU"] = np.zeros_like(z)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, D, yl in zip(axs, [f, d1, d2], [r"$\sigma(z)$", r"$\sigma'(z)$", r"$\sigma''(z)$"]):
        for k, v in D.items():
            c, ls, _ = STY[k]; ax.plot(z, v, color=c, ls=ls, lw=2.4, label=k)
        ax.set_xlabel("$z$"); ax.set_ylabel(yl); ax.axhline(0, color=GRAY, lw=0.6)
    axs[0].set_ylim(-1.2, 3.2); axs[0].legend(frameon=False, loc="upper left")
    axs[2].annotate(r"ReLU: $\delta(z)$ at 0", xy=(0, 0), xytext=(0.8, 0.45), color=RED, fontsize=15,
                    arrowprops=dict(arrowstyle="->", color=RED))
    axs[2].plot([0, 0], [0, 0.4], color=RED, lw=2.4)
    for ax, l in zip(axs, "abc"): panel_label(ax, l)
    fig.tight_layout(); fig.savefig("Fig7.1.1_activations.png"); plt.close(fig)

if "2" in which:   # ReLU vs smooth: forces, curvature, NVE energy error
    d = np.load(L + "71_activation.npz"); rg = d["rg"]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.7))
    m = (rg > 0.9) & (rg < 1.3)
    axs[0].plot(rg[m], d["F_true"][m], color=GRAY, lw=5, alpha=0.45, label="Morse (true)")
    for k in ["ReLU", "SiLU"]:
        c, ls, _ = STY[k]; axs[0].plot(rg[m], d[f"F_{k}"][m], color=c, ls=ls, lw=2, label=k)
    mt = (d["r_tr"] > 0.9) & (d["r_tr"] < 1.3)
    axs[0].plot(d["r_tr"][mt], d["F_tr"][mt], "o", ms=6, mfc="white", mec=NAVY, label="training points")
    axs[0].set_xlabel(r"$r$ (Å)"); axs[0].set_ylabel(r"$F=-dE/dr$ (eV/Å)"); axs[0].legend(frameon=False, fontsize=13)
    for k in ["ReLU", "GELU", "SiLU", "tanh"]:
        c, ls, _ = STY[k]; axs[1].plot(rg, d[f"k_{k}"], color=c, ls=ls, lw=2, label=k)
    axs[1].axhline(32.0, color=GRAY, lw=1); axs[1].plot(rg, 2 * 4.0 * 4.0 * (2 * np.exp(-4 * (rg - 1)) - np.exp(-2 * (rg - 1))),
                                                          color=GRAY, lw=5, alpha=0.45, label="Morse (true)")
    axs[1].set_ylim(-20, 90); axs[1].set_xlabel(r"$r$ (Å)"); axs[1].set_ylabel(r"$d^2E/dr^2$ (eV/Å$^2$)")
    axs[1].legend(frameon=False, fontsize=13, ncol=2)
    dts = d["dts"]; mk = {"ReLU": "o", "GELU": "s", "SiLU": "^", "tanh": "D"}
    for k in ["ReLU", "GELU", "SiLU", "tanh"]:
        c, ls, _ = STY[k]; axs[2].loglog(dts, 1e3 * d[f"drift_{k}"], color=c, ls=ls, marker=mk[k], ms=8, lw=2, label=k)
    axs[2].set_xlabel(r"time step $\Delta t$ (fs)"); axs[2].set_ylabel(r"max $|E_{\mathrm{tot}}-E_0|$ (meV)")
    axs[2].text(0.13, 30, r"slope 1", color=RED, fontsize=15); axs[2].text(0.16, 0.12, r"slope 2", color=NAVY, fontsize=15)
    axs[2].set_xticks(dts); axs[2].set_xticklabels([f"{v:g}" for v in dts]); axs[2].minorticks_off()
    axs[2].legend(frameon=False, fontsize=13, loc="lower right")
    for ax, l in zip(axs, "abc"): panel_label(ax, l)
    fig.tight_layout(); fig.savefig("Fig7.1.2_relu_forces.png"); plt.close(fig)

if "3" in which:   # gradient flow
    d = np.load(L + "71_backprop_gradflow.npz"); names, G = d["names"], d["G"]
    sty = [(GOLD, "-.", "v"), (GREEN, ":", "D"), (GRAY, "--", "x"), (PURPLE, "--", "+"), (RED, "-", "o"), (NAVY, "-", "s")]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for n, g, (c, ls, mk) in zip(names, G, sty):
        ax.semilogy(np.arange(1, len(g) + 1), g, color=c, ls=ls, marker=mk, ms=6, lw=2, label=str(n).replace("0.05^2", "$0.05^2$").replace("0.3^2", "$0.3^2$"))
    ax.set_xlabel("layer index $l$ (1 = input side)"); ax.set_ylabel(r"$\|\partial\mathcal{L}/\partial W_l\|_F$")
    ax.legend(frameon=False, fontsize=12.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig("Fig7.1.3_gradflow.png"); plt.close(fig)

if "4" in which:   # optimizers
    d = np.load(L + "71_optim.npz")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    sty = {"SGD": (GRAY, ":"), "SGD+momentum": (BLUE, "--"), "Nesterov": (GREEN, "-."), "Adagrad": (GOLD, "--"),
           "RMSProp": (PURPLE, ":"), "Adam": (RED, "-"), "AdamW(wd=1e-2)": (NAVY, "-")}
    for n, c in zip(d["names"], d["curves"]):
        col, ls = sty[str(n)]; axs[0].plot(np.arange(1, len(c) + 1), c, color=col, ls=ls, lw=2, label=str(n))
    axs[0].set_ylim(0.38, 1.02); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("test MAE (eV)")
    axs[0].legend(frameon=False, fontsize=12, ncol=2, loc="upper right")
    t = d["toy"]; x = np.where(t[:, 0] == 0, 1e-4, t[:, 0])
    axs[1].semilogx(x, t[:, 1], "o-", color=RED, lw=2, ms=8, label="Adam + L2")
    axs[1].semilogx(x, t[:, 2], "s--", color=NAVY, lw=2, ms=8, label="AdamW")
    axs[1].set_xlabel(r"gradient-noise scale $\sigma$ (0 plotted at $10^{-4}$)"); axs[1].set_ylabel(r"final $|\theta|$")
    axs[1].legend(frameon=False)
    lam = d["lams"]
    axs[2].semilogx(lam, d["wd_adam"][:, 0], "o-", color=RED, lw=2, ms=8, label="Adam + L2")
    axs[2].semilogx(lam, d["wd_adamw"][:, 0], "s--", color=NAVY, lw=2, ms=8, label="AdamW")
    axs[2].set_xlabel(r"$\lambda$"); axs[2].set_ylabel("test MAE (eV)"); axs[2].legend(frameon=False)
    for ax, l in zip(axs, "abc"): panel_label(ax, l)
    fig.tight_layout(); fig.savefig("Fig7.1.4_optimizers.png"); plt.close(fig)

if "5" in which:   # double descent
    d = np.load(L + "71_double_descent.npz"); N, n = d["N"], int(d["n"])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axs[0].loglog(N, d["te"], "o-", color=RED, lw=2, ms=7, label="test, min-norm")
    axs[0].loglog(N, d["ridge"], "s--", color=NAVY, lw=2, ms=7, label=r"test, ridge $\lambda=10$")
    axs[0].loglog(N, np.maximum(d["tr"], 1e-3), "^:", color=GRAY, lw=2, ms=7, label="train, min-norm")
    axs[0].axvline(n, color=GOLD, lw=2); axs[0].text(n * 1.1, 30, r"$N=n$", color=GOLD, fontsize=16)
    axs[0].set_xlabel("number of random features $N$"); axs[0].set_ylabel("RMSE (eV)"); axs[0].legend(frameon=False, fontsize=13)
    axs[1].loglog(N, d["w"], "o-", color=PURPLE, lw=2, ms=7); axs[1].axvline(n, color=GOLD, lw=2)
    axs[1].set_xlabel("number of random features $N$"); axs[1].set_ylabel(r"$\|\mathbf{w}\|_2$ (min-norm)")
    for ax, l in zip(axs, "ab"): panel_label(ax, l)
    fig.tight_layout(); fig.savefig("Fig7.1.5_double_descent.png"); plt.close(fig)
print("done", which)
