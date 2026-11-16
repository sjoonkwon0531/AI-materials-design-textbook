"""Fig5.1.1: multiscale methods map (schematic order-of-magnitude ranges)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from bookstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.5, 7.6))
# (log10 length [m] min, max, log10 time [s] min, max, label, color, label position)
boxes = [
    (-11.1, -8.5, -16.9, -15.3, "DFT (electrons)", NAVY, (-9.8, -16.1)),
    (-10.4, -8.8, -15.0, -10.6, "AIMD", PURPLE, (-9.6, -11.25)),
    (-10.1, -6.9, -15.0, -5.5, "classical MD", BLUE, (-7.75, -6.25)),
    (-10.2, -7.4, -15.0, -8.0, "MLIP-MD", GREEN, (-8.3, -8.75)),
    (-9.0, -5.6, -12.2, -3.5, "CG-MD / DPD", GOLD, (-6.6, -4.25)),
    (-9.6, -6.1, -9.5, 1.0, "KMC", RED, (-8.6, 0.25)),
    (-8.0, -3.0, -6.0, 3.0, "phase-field", GRAY, (-4.2, 2.25)),
    (-6.5, 0.3, -6.5, 5.0, "FEM / LBM (continuum)", NAVY, (-1.9, 4.25)),
]
for x0, x1, y0, y1, lab, c, (tx, ty) in boxes:
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc=c, ec="none", alpha=0.10))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="none", ec=c, lw=2.2, ls="--" if lab == "MLIP-MD" else "-"))
    ax.text(tx, ty, lab, ha="center", va="center", fontsize=18, fontweight="bold",
            color="#9A6B00" if c == GOLD else c,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

def arrow(x0, y0, x1, y1, text, tx, ty, rad=-0.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=24, lw=2.4,
                                 color="black", connectionstyle=f"arc3,rad={rad}", zorder=5))
    ax.text(tx, ty, text, fontsize=15, ha="center", va="center", style="italic", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0))

arrow(-8.6, -15.8, -7.9, -13.4, "training data", -7.2, -14.4, rad=0.25)
arrow(-10.75, -15.3, -9.4, -0.6, "barriers → rates", -10.9, -3.0, rad=-0.12)
arrow(-7.3, -12.0, -6.0, -9.6, "IBI / force matching", -5.0, -11.2, rad=0.2)
arrow(-6.3, -5.0, -5.4, -1.0, "mobility, interfaces", -4.4, -3.4, rad=0.2)
arrow(-4.0, 0.6, -2.2, 2.6, "homogenization", -1.6, 0.9, rad=-0.2)

ax.set_xlim(-11.7, 0.8)
ax.set_ylim(-17, 5.6)
ax.set_xlabel(r"length scale  $\log_{10}(\ell\,/\,\mathrm{m})$")
ax.set_ylabel(r"time scale  $\log_{10}(t\,/\,\mathrm{s})$")
ax.set_xticks(range(-11, 1))
ax.set_yticks(range(-16, 6, 2))
# reference ticks on top/right in familiar units
for x, lab in [(-10, "1 Å"), (-9, "1 nm"), (-6, "1 μm"), (-3, "1 mm"), (0, "1 m")]:
    ax.text(x, 5.75, lab, ha="center", va="bottom", fontsize=14, color=GRAY)
for y, lab in [(-15, "1 fs"), (-12, "1 ps"), (-9, "1 ns"), (-6, "1 μs"), (-3, "1 ms"), (0, "1 s")]:
    ax.axhline(y, color=GRAY, lw=0.6, ls=":", zorder=0)
    ax.text(0.95, y, lab, ha="left", va="center", fontsize=14, color=GRAY)
fig.savefig("../../ch05/figures/Fig5.1.1_multiscale_map.png")
print("saved")
