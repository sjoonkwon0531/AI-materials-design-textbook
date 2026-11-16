"""Shared figure style for the textbook (new figure rules).
Arial Narrow (Liberation Sans Narrow is the metric-compatible fallback here),
large fonts, panel labels a/b/c only, no panel titles, English labels,
Midnight Executive palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY="#1E2761"; ICE="#CADCFC"; GOLD="#F4B942"; RED="#C0392B"
GREEN="#1E8449"; PURPLE="#7D3C98"; BLUE="#2471A3"; GRAY="#6B6B6B"
CYCLE=[NAVY, RED, BLUE, GOLD, GREEN, PURPLE]

plt.rcParams.update({
    "font.family": ["Arial Narrow", "Liberation Sans Narrow"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans Narrow",
    "mathtext.it": "Liberation Sans Narrow:italic",
    "mathtext.bf": "Liberation Sans Narrow:bold",
    "font.size": 17, "axes.labelsize": 19,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    "axes.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

def panel_label(ax, letter, x=-0.14, y=1.02):
    """Bold lowercase panel label (a, b, c) at the top-left; never a title."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="bottom", ha="left")
