"""Fig6.1.1 -- descriptor families vs target properties (qualitative map)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *  # noqa
import numpy as np
from matplotlib.patches import Rectangle

fam = ["Empirical scalars\n(t, τ, Δχ, Mendeleev no.)",
       "Composition statistics\n(Magpie)",
       "Global matrices\n(CM, sine, Ewald)",
       "Global k-body histograms\n(MBTR)",
       "Local atom-centered\n(ACSF, SOAP)",
       "Learned representations\n(GNN; Ch.7)"]
tgt = ["Structure-type\nstability", "Band gap\n(experiment)", "Formation\nenergy",
       "Molecular\natomization E", "PES and forces\n(MLIP)", "Similarity /\nstructure search"]
# 2 = primary use, 1 = used / possible, 0 = not suited
U = np.array([
    [2, 1, 1, 0, 0, 0],
    [1, 2, 2, 0, 0, 0],
    [0, 0, 1, 2, 0, 1],
    [0, 0, 2, 2, 0, 1],
    [0, 0, 1, 2, 2, 2],
    [1, 1, 2, 2, 2, 1],
])
inp = ["composition", "composition", "structure", "structure", "structure", "structure"]

fig, ax = plt.subplots(figsize=(11.5, 7.2))
for i in range(len(fam)):
    col = ICE if inp[i] == "composition" else "#F2F2F2"
    ax.add_patch(Rectangle((-0.5, i - 0.5), len(tgt), 1, color=col, zorder=0, lw=0))
    for j in range(len(tgt)):
        if U[i, j] == 2:
            ax.plot(j, i, "o", ms=24, color=NAVY, zorder=3)
        elif U[i, j] == 1:
            ax.plot(j, i, "o", ms=20, mfc="white", mec=BLUE, mew=2.5, zorder=3)
        else:
            ax.plot(j, i, "x", ms=10, color=GRAY, mew=1.5, zorder=3)
ax.set_xlim(-0.5, len(tgt) - 0.5); ax.set_ylim(len(fam) - 0.5, -0.5)
ax.set_xticks(range(len(tgt))); ax.set_xticklabels(tgt, fontsize=15)
ax.xaxis.tick_top()
ax.set_yticks(range(len(fam))); ax.set_yticklabels(fam, fontsize=15)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.text(len(tgt) - 0.45, 0.5, "composition\nonly", rotation=90, va="center", ha="left",
        fontsize=15, color=NAVY)
ax.text(len(tgt) - 0.45, 3.5, "atomic\nstructure", rotation=90, va="center", ha="left",
        fontsize=15, color=GRAY)
h1, = ax.plot([], [], "o", ms=14, color=NAVY, label="primary use")
h2, = ax.plot([], [], "o", ms=13, mfc="white", mec=BLUE, mew=2.2, label="used / possible")
h3, = ax.plot([], [], "x", ms=10, color=GRAY, mew=1.5, label="not suited")
ax.legend(handles=[h1, h2, h3], loc="upper center", bbox_to_anchor=(0.45, -0.02), ncol=3,
          frameon=False)
fig.savefig("../ch06/figures/Fig6.1.1_descriptor_map.png")
print("saved")
