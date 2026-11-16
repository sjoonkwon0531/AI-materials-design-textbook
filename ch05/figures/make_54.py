"""Fig5.4.1: iterative Boltzmann inversion for 2D LJ dimers -> single beads."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
d = np.load("../ch05/labs/54_ibi_fig.npz")
r = d["r"]; m = r > 0.6
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))
a = ax[0, 0]
a.plot(r[m], d["g_t"][m], color=GRAY, lw=4.5, alpha=0.6, label="target (fine MD)")
a.plot(r[m], d["gs"][0][m], color=GOLD, lw=2, ls="--", label="iteration 0 (PMF guess)")
a.plot(r[m], d["gs"][-1][m], color=NAVY, lw=1.8, label="iteration 8 (IBI)")
a.set_xlabel("r"); a.set_ylabel("g(r)"); a.set_xlim(0.6, 3.5)
a.legend(frameon=False, loc="lower right")
panel_label(a, "a")
a = ax[0, 1]
Vs = d["Vs"]
a.plot(r, Vs[0], color=GOLD, lw=2, ls="--", label="$V_0=-k_BT\\,\\ln g$")
a.plot(r, Vs[-1], color=NAVY, lw=2.2, label="IBI")
a.plot(r, d["Vp"], color=RED, lw=2, ls="-.", label="IBI + pressure corr.")
a.axhline(0, color=GRAY, lw=0.8)
a.set_xlim(0.6, 3.5); a.set_ylim(-0.8, 2.0)
a.set_xlabel("r"); a.set_ylabel("V(r)  ($k_BT$)")
a.legend(frameon=False, loc="upper right")
panel_label(a, "b")
h = d["hist"]; hp = d["hp"]
n1 = h.shape[0]
it_all = np.concatenate([h[:, 0], n1 + hp[:, 0]])
a = ax[1, 0]
a.semilogy(h[:, 0], h[:, 1], "o-", color=NAVY, ms=8, label="IBI")
a.semilogy(n1 + hp[:, 0], hp[:, 1], "s--", color=RED, ms=8, label="IBI + pressure corr.")
a.axhline(0.0194, color=GRAY, ls=":", lw=2, label="statistical noise floor")
a.set_xlabel("iteration"); a.set_ylabel("g(r) RMSE")
a.legend(frameon=False, loc="upper right")
panel_label(a, "c")
a = ax[1, 1]
a.plot(h[:, 0], h[:, 2], "o-", color=NAVY, ms=8, label="IBI")
a.plot(n1 + hp[:, 0], hp[:, 2], "s--", color=RED, ms=8, label="IBI + pressure corr.")
a.axhline(float(d["P_fine"]), color=GREEN, lw=2.2, ls=":", label="fine MD")
a.set_xlabel("iteration"); a.set_ylabel("pressure P")
a.set_ylim(0.2, 0.7)
a.legend(frameon=False, loc="upper right")
panel_label(a, "d")
for a in (ax[1, 0], ax[1, 1]):
    a.set_xticks(range(0, 16, 3))
fig.tight_layout(h_pad=2.0, w_pad=2.5)
fig.savefig("../ch05/figures/Fig5.4.1_ibi.png")
