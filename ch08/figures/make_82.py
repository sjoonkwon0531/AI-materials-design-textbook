"""Figures for Sec.8.2 (Fig8.2.1-Fig8.2.3). Run from ch8/figs after the labs have produced their npz/json."""
import sys, os, json, importlib
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = os.path.join(HERE, "..", "labs")
sys.path.insert(0, LABS)

# ---------------- Fig 8.2.1: composition environment and reward hacking ----------------
z = np.load(os.path.join(LABS, "82_comp_rl.npz"))
mu, sg, second, bal, tie, St = z["mu"], z["sigma"], z["second"], z["balanced"], z["tie"], z["states"]
m = importlib.import_module("82_compenv")
tab = m.GridTable(("Cu", "In", "Ga", "S", "Se"), 12, m.train_surrogate())
env = m.CompositionEnv(tab, 1.4, 8)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
a = ax[0]
for (nIn, nGa), col, mk, lab in [((3, 0), NAVY, "o", "Cu$_3$In$_3$X$_6$"), ((1, 2), RED, "s", "Cu$_3$InGa$_2$X$_6$"),
                                 ((0, 3), BLUE, "^", "Cu$_3$Ga$_3$X$_6$")]:
    xs, ys, es, ts = [], [], [], []
    for i, s in enumerate(St):
        if s[0] == 3 and s[1] == nIn and s[2] == nGa:
            xs.append(s[3] / 6); ys.append(mu[i]); es.append(sg[i]); ts.append(tie[i])
    o = np.argsort(xs); xs, ys, es, ts = map(lambda v: np.array(v)[o], (xs, ys, es, ts))
    a.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=8, lw=1.8, capsize=3, label=lab)
    a.scatter(xs[ts], ys[ts], s=260, facecolors="none", edgecolors=GOLD, linewidths=2.5, zorder=5)
a.axhline(1.4, color=GRAY, ls="--", lw=1.2)
a.set_xlabel("S fraction  S/(S+Se)"); a.set_ylabel("predicted $E_g$ (eV)")
a.set_ylim(0.85, 2.75); a.legend(loc="upper center", frameon=False, fontsize=13, ncol=2); panel_label(a, "a")
a = ax[1]
a.scatter(mu[~bal], sg[~bal], s=8, color=GRAY, alpha=0.35, label="not charge-balanced")
a.scatter(mu[bal], sg[bal], s=40, color=NAVY, marker="o", label="charge-balanced")
for key, col, mk, lab in [("idx_0", RED, "x", "Q end points: naive"), ("idx_3", GREEN, "D", "Q end points: balance + $\\sigma$")]:
    u = np.unique(z[key]); a.scatter(mu[u], sg[u], s=90, color=col, marker=mk, linewidths=2.2, label=lab, zorder=4,
                                     facecolors="none" if mk == "D" else None)
a.axvline(1.4, color=GRAY, ls="--", lw=1.2)
a.set_xlabel("ensemble mean $\\mu$ (eV)"); a.set_ylabel("ensemble std $\\sigma$ (eV)")
a.set_ylim(-0.01, 0.47); a.legend(loc="upper center", frameon=False, fontsize=12.5, ncol=2, columnspacing=0.8, handletextpad=0.3); panel_label(a, "b")
a = ax[2]
S = tab.score(1.4); rng = np.random.default_rng(1)
starts = [tab.states[i] for i in rng.choice(len(tab.states), 200, replace=False)][:100]
def run(s, pol):
    for _ in range(8): s = env.next_state(s, pol(s))
    return S[tab.index[s]]
greedy = lambda s: max(range(21), key=lambda a_: S[tab.index[env.next_state(s, a_)]])
g = np.mean([run(s, greedy) for s in starts]); r0 = np.mean([S[tab.index[s]] for s in starts])
ep = np.arange(1, 21) * 1000
a.plot(ep, -z["curve_0"], color=RED, marker="o", lw=2, label="Q-learning (greedy policy)")
a.axhline(-g, color=BLUE, ls="--", lw=2, label="greedy, 21 queries/step")
a.axhline(-r0, color=GRAY, ls=":", lw=2, label="start compositions")
a.set_yscale("log"); a.set_xlabel("training episodes"); a.set_ylabel("mean $|\\mu-1.4|$ after 8 edits (eV)")
a.legend(loc="center right", frameon=False, fontsize=13); panel_label(a, "c")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.2.1_comp_rl.png")); plt.close(fig)
print("Fig8.2.1: greedy mean |mu-1.4| on 100 starts", round(g, 4), "start", round(r0, 3),
      "Q curve first/last", np.round(-z["curve_0"][[0, -1]], 4))

# ---------------- Fig 8.2.2: MCTS vs random rollouts ----------------
J = json.load(open(os.path.join(LABS, "82_mcts.json")))
B = [10, 30, 100, 300, 1000]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for name, col, mk, ls in [("random rollouts", GRAY, "s", "--"), ("MCTS (UCT c=0.4)", NAVY, "o", "-")]:
    v = np.array([J[f"{name}_{b}"] for b in B])
    ax[0].errorbar(B, v[:, 1], yerr=v[:, 4], color=col, marker=mk, ls=ls, lw=2, ms=8, capsize=3, label=name.replace(" (UCT c=0.4)", " (UCT)"))
    ax[1].plot(B, v[:, 2], color=col, marker=mk, ls=ls, lw=2, ms=8, label=name.replace(" (UCT c=0.4)", " (UCT)"))
ax[0].axhline(J["greedy"], color=RED, ls=":", lw=2, label="greedy (highest yield)")
ax[0].set_xscale("log"); ax[1].set_xscale("log")
ax[0].set_xlabel("simulations"); ax[1].set_xlabel("simulations")
ax[0].set_ylabel("route value / DP optimum"); ax[1].set_ylabel("fraction of targets solved optimally")
ax[0].legend(frameon=False, loc="upper left"); ax[0].set_ylim(0, 1); ax[1].set_ylim(0, 1)
panel_label(ax[0], "a"); panel_label(ax[1], "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.2.2_mcts.png")); plt.close(fig)

# ---------------- Fig 8.2.3: re-entrant fab ----------------
F = np.load(os.path.join(LABS, "82_fab.npz"))
fig = plt.figure(figsize=(17, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0])
a = fig.add_subplot(gs[0]); a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
pos = {"dep": (2, 5), "litho": (5, 5), "etch": (8, 5)}
for k, (x, y_) in pos.items():
    a.add_patch(plt.Rectangle((x - 1.1, y_ - 0.8), 2.2, 1.6, color=[BLUE, RED, GREEN][list(pos).index(k)], alpha=0.85))
    a.text(x, y_, k, ha="center", va="center", color="white", fontsize=18, fontweight="bold")
for layer, yy in enumerate([7.8, 8.9, 9.9]):
    pass
a.annotate("", xy=(3.9, 5.3), xytext=(3.1, 5.3), arrowprops=dict(arrowstyle="->", lw=2, color=NAVY))
a.annotate("", xy=(6.9, 5.3), xytext=(6.1, 5.3), arrowprops=dict(arrowstyle="->", lw=2, color=NAVY))
a.annotate("", xy=(2, 4.1), xytext=(8, 4.1), arrowprops=dict(arrowstyle="->", lw=2, color=NAVY, connectionstyle="arc3,rad=-0.45"))
a.text(5, 1.2, "re-enter: layer 1 $\\to$ 2 $\\to$ 3", ha="center", fontsize=16, color=NAVY)
a.text(2, 6.3, "lots in", ha="center", fontsize=15); a.annotate("", xy=(2, 5.85), xytext=(2, 6.25), arrowprops=dict(arrowstyle="->", lw=1.5))
a.text(5, 8.6, "litho load 3.2 h/lot (bottleneck)", ha="center", fontsize=15, color=RED)
panel_label(a, "a", x=0.0, y=0.95)
a = fig.add_subplot(gs[1])
pol = ["FBFS", "FIFO", "SPT", "random", "FIFO (fab)", "least slack", "RL seed0 (greedy)", "RL seed1 (greedy)",
       "RL seed2 (greedy)", "RL from least-slack (greedy)"]
lab = ["FBFS", "FIFO (queue)", "SPT", "random", "FIFO (fab) = LBFS", "least slack", "RL seed 0", "RL seed 1",
       "RL seed 2", "RL from least slack"]
yy = np.arange(len(pol))
for off, rate, col, mk in [(-0.18, "0.2875", NAVY, "o"), (0.18, "0.3", RED, "s")]:
    w = np.array([F[f"{rate}_{p}_wip"].mean() for p in pol]); e = np.array([F[f"{rate}_{p}_wip"].std() / np.sqrt(30) for p in pol])
    a.errorbar(w, yy + off, xerr=e, fmt=mk, color=col, ms=8, capsize=3,
               label=f"litho utilization {0.92 if rate == '0.2875' else 0.96:.2f}")
a.set_yticks(yy); a.set_yticklabels(lab, fontsize=14); a.invert_yaxis(); a.set_xscale("log")
a.set_xticks([6, 8, 10, 15, 20, 30]); a.set_xticklabels(["6", "8", "10", "15", "20", "30"])
a.set_xlabel("time-averaged WIP (lots)"); a.legend(frameon=False, loc="lower right", fontsize=13)
panel_label(a, "b", x=-0.42)
a = fig.add_subplot(gs[2])
for k, col, ls in [("RL seed0", NAVY, "-"), ("RL seed1", BLUE, "--"), ("RL seed2", PURPLE, "-."), ("RL from least-slack", GOLD, ":")]:
    h = F["hist_" + k]; sm = np.convolve(-h, np.ones(10) / 10, mode="valid")
    a.plot(np.arange(len(sm)) + 10, sm, color=col, ls=ls, lw=2.2, label=k.replace("RL ", "").replace("from least-slack", "init: least slack"))
a.set_xlabel("REINFORCE iteration"); a.set_ylabel("training WIP (lots, 10-it. mean)")
a.set_ylim(4.7, 9.2); a.legend(frameon=False, fontsize=13, ncol=2, loc="upper center"); panel_label(a, "c")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.2.3_fab.png")); plt.close(fig)
print("done")
