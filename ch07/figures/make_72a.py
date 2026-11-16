"""Figures for sec 7.2.1-7.2.2 (run from ch7/figs). Reads lab npz files in ../labs."""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import matplotlib.pyplot as plt
L = "../labs/"
which = sys.argv[1:] or ["1", "2", "3"]

if "1" in which:   # XRD: example patterns + accuracy vs zero shift
    d = np.load(L + "72a_xrd_cnn.npz", allow_pickle=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    names = ["fcc", "rocksalt", "zincblende", "fluorite", "rutile", "perovskite"]
    allnames = ["fcc", "bcc", "hcp", "rocksalt", "CsCl", "zincblende", "wurtzite", "fluorite", "rutile", "perovskite"]
    for k, n in enumerate(names):
        y = d["ex"][allnames.index(n)]
        axs[0].plot(d["tt"], y / y.max() + 1.15 * k, color=[NAVY, RED, BLUE, GREEN, PURPLE, GOLD][k], lw=1.3)
        axs[0].text(80.5, 1.15 * k + 0.25, n, fontsize=14, va="center")
    axs[0].set_xlim(10, 80); axs[0].set_yticks([]); axs[0].set_xlabel(r"$2\theta$ (deg, Cu K$\alpha$)")
    axs[0].set_ylabel("intensity (offset)")
    sty = {"MLP": (GRAY, "--", "o"), "CNN+flatten": (BLUE, ":", "s"), "CNN+global pool": (GOLD, "-", "^"), "deep CNN+global pool": (NAVY, "-", "v"),
           "MLP + shift augmentation": (RED, "-.", "D")}
    for n, a in zip(d["names"], d["acc"]):
        c, ls, mk = sty[str(n)]
        axs[1].plot(d["shifts"], a, color=c, ls=ls, marker=mk, ms=8, lw=2, label=str(n))
    axs[1].axhline(0.1, color=GRAY, lw=0.8); axs[1].text(1.3, 0.12, "chance", color=GRAY, fontsize=13)
    axs[1].set_xlabel(r"zero shift of test patterns (deg)"); axs[1].set_ylabel("test accuracy")
    axs[1].set_ylim(0, 1.02); axs[1].legend(frameon=False, fontsize=12.5, loc="lower right", bbox_to_anchor=(1.0, 0.13))
    panel_label(axs[0], "a", x=-0.05); panel_label(axs[1], "b")
    fig.tight_layout(); fig.savefig("Fig7.2.1_xrd_cnn.png"); plt.close(fig)

if "2" in which:   # microstructures + rotation failure
    d = np.load(L + "72a_micro_cnn.npz")
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.15, 2.6])
    titles = ["lamellar, 0 deg", "lamellar, 60 deg", "labyrinth", "droplets"]
    for k in range(4):
        ax = fig.add_subplot(gs[k // 2, k % 2]); ax.imshow(d["ex"][k], cmap="gray_r", vmin=-0.3, vmax=1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(titles[k], fontsize=13)
        if k == 0: panel_label(ax, "a", x=-0.12, y=1.04)
    ax = fig.add_subplot(gs[:, 3])
    ax.axvspan(0, 15, color=GOLD, alpha=0.25, lw=0); ax.text(1, 0.05, "training\nangles", fontsize=13)
    ax.plot(d["angles"], d["noaug"][:, 0], "o-", color=RED, lw=2, ms=8, label="lamellae, no augmentation")
    ax.plot(d["angles"], d["aug"][:, 0], "s--", color=NAVY, lw=2, ms=8, label="lamellae, rotation augmentation")
    ax.plot(d["angles"], d["noaug"][:, 1], "^:", color=GRAY, lw=2, ms=8, label="isotropic classes, no augmentation")
    ax.set_xlabel("lamella orientation in test images (deg)"); ax.set_ylabel("recall"); ax.set_ylim(0, 1.05)
    ax.set_xticks(d["angles"]); ax.legend(frameon=False, fontsize=13, loc="center right")
    panel_label(ax, "b", x=-0.1)
    fig.tight_layout(); fig.savefig("Fig7.2.2_micro_rotation.png"); plt.close(fig)

if "3" in which:   # RNN/LSTM gradients + battery forecast
    d = np.load(L + "72a_lstm.npz", allow_pickle=True)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.9))
    lag = np.arange(len(d["lagG_rnn"]))
    for key, c, ls, lab in [("lagG_rnn", RED, "-", "vanilla RNN (tanh)"), ("lagG_lstm0", BLUE, "--", "LSTM, forget bias 0"),
                            ("lagG_lstm3", NAVY, "-", "LSTM, forget bias +3")]:
        axs[0].semilogy(lag[1:], d[key][1:], color=c, ls=ls, lw=2.2, label=lab)
    axs[0].set_xlabel(r"lag $T-t$ (steps)"); axs[0].set_ylabel(r"$\|\partial h_T/\partial x_t\|$"); axs[0].legend(frameon=False, fontsize=13)
    q, s, nk = d["ex_q"], int(d["ex_s"]), float(d["ex_nk"])
    n = np.arange(len(q)); W = 50; hz = np.arange(s + W, s + W + 100)
    axs[1].plot(n, 100 * q, color=GRAY, lw=1, label="synthetic cell")
    axs[1].axvspan(s, s + W - 1, color=GOLD, alpha=0.3, lw=0)
    axs[1].plot(hz, 100 * d["ex_linear"], color=GREEN, ls=":", lw=2.4, label="linear extrapolation")
    axs[1].plot(hz, 100 * d["ex_MLP"], color=BLUE, ls="--", lw=2.4, label="MLP")
    axs[1].plot(hz, 100 * d["ex_LSTM"], color=RED, ls="-", lw=2.4, label="LSTM")
    axs[1].axvline(nk, color=PURPLE, lw=1.2); axs[1].text(nk + 6, 0.93, "knee", color=PURPLE, fontsize=14, transform=axs[1].get_xaxis_transform())
    axs[1].set_xlim(s - 250, s + W + 160); axs[1].set_ylim(100 * q[min(s + W + 160, len(q) - 1)] - 3, 100 * q[max(s - 250, 0)] + 2)
    axs[1].set_xlabel("cycle"); axs[1].set_ylabel("capacity (% of nominal)"); axs[1].legend(frameon=False, fontsize=12, loc="lower left")
    sty = {"persistence": (GRAY, "-."), "linear extrapolation": (GREEN, ":"), "quadratic extrapolation": (GOLD, "-"),
           "MLP": (BLUE, "--"), "LSTM": (RED, "-")}
    for nm, r in zip(d["names"], d["byh"]):
        c, ls = sty[str(nm)]; axs[2].plot(np.arange(1, 101), r, color=c, ls=ls, lw=2.2, label=str(nm))
    axs[2].set_xlabel("forecast horizon (cycles)"); axs[2].set_ylabel("test RMSE (% of nominal)"); axs[2].legend(frameon=False, fontsize=12)
    for ax, l in zip(axs, "abc"): panel_label(ax, l)
    fig.tight_layout(); fig.savefig("Fig7.2.3_lstm_battery.png"); plt.close(fig)
print("done", which)
