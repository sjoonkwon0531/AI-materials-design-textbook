"""Figures for §8.1 (Fig8.1.1-8.1.4). Reads data written by labs/81_*.py."""
import sys, importlib, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); sys.path.insert(0, "../ch08/labs")
from bookstyle import *
D = "../ch08/data/"; F = "../ch08/figures/"
which = sys.argv[1:] or ["bandit", "anneal", "cliff", "pg"]

def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")

if "bandit" in which:
    d = np.load(D + "81_bandit.npz")
    names = [("greedy_(eps=0)", "greedy", NAVY, "-"), ("eps-greedy_0.01", r"$\varepsilon$-greedy 0.01", BLUE, "--"),
             ("eps-greedy_0.1", r"$\varepsilon$-greedy 0.1", RED, "-."), ("UCB_c=2", "UCB, c = 2", GREEN, ":")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, lab, c, ls in names:
        rew, opt, creg = d[k]
        ax[0].plot(np.arange(1, 1001), creg, color=c, ls=ls, lw=2.4, label=lab)
        ax[1].plot(np.arange(10, 1001), 100 * smooth(opt, 10), color=c, ls=ls, lw=2.2, label=lab)
    ax[0].set_xlabel("Pull (experiment) number"); ax[0].set_ylabel("Cumulative regret")
    ax[1].set_xlabel("Pull (experiment) number"); ax[1].set_ylabel("Optimal arm chosen (%)")
    ax[0].legend(frameon=False, loc="upper left"); ax[1].set_ylim(0, 100)
    panel_label(ax[0], "a"); panel_label(ax[1], "b")
    fig.tight_layout(); fig.savefig(F + "Fig8.1.1_bandit.png"); plt.close(fig)

if "anneal" in which:
    E = importlib.import_module("81_anneal_env"); L = importlib.import_module("81_anneal")
    d = np.load(D + "81_anneal.npz"); Vs, pis = d["Vs"], d["pis"]
    V = np.zeros(E.nS); err = []
    for k in range(400):
        V = L.q_from_v(V).max(1); err.append(np.abs(V - Vs).max())
    err = np.array(err); k = np.arange(1, 401)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].semilogy(k, err, color=NAVY, lw=2.6, label=r"$\|V_k - V^*\|_\infty$")
    ax[0].semilogy(k, err[0] * E.GAMMA ** (k - 1), color=RED, ls="--", lw=2.0, label=r"$\gamma^k$ bound slope")
    ax[0].set_xlabel("Value-iteration sweep $k$"); ax[0].set_ylabel("Max error"); ax[0].legend(frameon=False)
    n = np.arange(E.nS)
    ax[1].step(n, E.TEMPS[pis], where="mid", color=RED, lw=2.6)
    ax[1].set_xlabel("Defects $n$"); ax[1].set_ylabel(r"Optimal $T^*(n)$ (K)")
    ax[1].set_yticks(E.TEMPS)
    tr = np.load(D + "81_anneal_traj.npy")
    t = np.arange(tr.shape[1])
    m = tr[:, :, 0].mean(0); s = tr[:, :, 0].std(0)
    ax[2].plot(t, m, color=NAVY, lw=2.6, label="defects $n$ (mean)")
    ax[2].fill_between(t, m - s, m + s, color=NAVY, alpha=0.15)
    ax[2].set_xlabel("Time step"); ax[2].set_ylabel("Defects $n$")
    a2 = ax[2].twinx(); a2.plot(t, tr[:, :, 1].mean(0), color=RED, ls="--", lw=2.4, label="mean $T$")
    a2.set_ylabel("Mean temperature (K)", color=RED); a2.spines["right"].set_visible(True)
    a2.tick_params(axis="y", colors=RED)
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, frameon=False, loc="lower left")
    for a, lab in zip(ax, "abc"): panel_label(a, lab)
    fig.tight_layout(); fig.savefig(F + "Fig8.1.2_anneal_dp.png"); plt.close(fig)

if "cliff" in which:
    d = np.load(D + "81_cliff_triad.npz")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for key, lab, c, ls in (("sarsa", "SARSA (on-policy)", BLUE, "-"), ("qlearn", "Q-learning (off-policy)", RED, "--")):
        ax[0].plot(np.arange(10, 501), smooth(d[key].mean(0), 10), color=c, ls=ls, lw=2.4, label=lab)
    ax[0].set_ylim(-110, 0); ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Online return per episode")
    ax[0].legend(frameon=False, loc="lower right")
    for key, lab, c, ls in (("baird_triad_vmax", "FA + bootstrap + off-policy", RED, "-"),
                            ("baird_mc_vmax", "no bootstrapping (MC target)", GREEN, "--"),
                            ("baird_tabular_vmax", "no FA (tabular)", NAVY, "-.")):
        ax[1].semilogy(np.arange(1, 5001), d[key], color=c, ls=ls, lw=2.4, label=lab)
    ax[1].set_xlabel("Expected-update sweep"); ax[1].set_ylabel(r"$\max_s |\hat v(s) - v_\pi(s)|$")
    ax[1].set_ylim(1e-6, 1e11); ax[1].legend(frameon=False, loc="upper left")
    panel_label(ax[0], "a"); panel_label(ax[1], "b")
    fig.tight_layout(); fig.savefig(F + "Fig8.1.3_cliff_triad.png"); plt.close(fig)

if "pg" in which:
    d = np.load(D + "81_pg_curves.npz"); v = np.load(D + "81_pg_variance.npy")
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for key, lab, c, ls in (("reinforce", "REINFORCE", GRAY, ":"), ("baseline", "REINFORCE + $V$ baseline", BLUE, "--"),
                            ("a2c", "A2C (GAE, $\\lambda$=0.95)", RED, "-")):
        L = d[key]; sm = np.array([smooth(l, 10) for l in L]); x = np.arange(10, L.shape[1] + 1)
        m = sm.mean(0); se = sm.std(0, ddof=1) / np.sqrt(len(sm))
        ax[0].plot(x, m, color=c, ls=ls, lw=2.5, label=lab); ax[0].fill_between(x, m - se, m + se, color=c, alpha=0.15)
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Episode length (return)"); ax[0].set_ylim(0, 660); ax[0].set_yticks(range(0, 501, 100))
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.02), ncol=3, fontsize=13, columnspacing=1.0, handlelength=2.2)
    labels = ["$G_0$", "$\\gamma^t G_t$", "$G_t$", "$G_t-b$", "$G_t-V$", "GAE .95", "TD $\\delta_t$"]
    cols = [GRAY, GOLD, NAVY, BLUE, GREEN, RED, PURPLE]
    ax[1].bar(range(7), v[:, 0] / v[2, 0], color=cols, edgecolor="black", linewidth=0.8)
    for i in range(7):
        ax[1].text(i, v[i, 0] / v[2, 0] * 1.25, f"{v[i,3]:.2f}", ha="center", fontsize=13)
    ax[1].set_ylim(3e-3, 8)
    ax[1].set_yscale("log"); ax[1].set_xticks(range(7)); ax[1].set_xticklabels(labels, rotation=30)
    ax[1].set_ylabel("Variance relative to $G_t$"); ax[1].axhline(1, color="black", lw=1)
    panel_label(ax[0], "a"); panel_label(ax[1], "b")
    fig.tight_layout(); fig.savefig(F + "Fig8.1.4_policy_gradient.png"); plt.close(fig)
print("done", which)
