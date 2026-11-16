"""Fig 7.4.1: pretraining vs scratch on ethanol GFN2-xTB 300 K, short (800) and long (2,300) fine-tuning.
Numbers parsed from labs/74_pretrain_ethanol.out, 74_scratch_control.out, 74_long_finetune.out."""
import sys, re, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import matplotlib.pyplot as plt
LAB = "../../ch07/labs/"
pat = re.compile(r"N=\s*(\d+) (?:init=(\w)|scratch)[^:]*: E MAE\s+[\d.]+ \(([\d.]+), ([\d.]+)\) meV\s+F MAE\s+[\d.]+ \(([\d.]+), ([\d.]+)\)")
data = {}
for fn, steps in (("74_pretrain_ethanol.out", 800), ("74_scratch_control.out", 2300), ("74_long_finetune.out", 2300)):
    for m in pat.finditer(open(LAB + fn).read()):
        N, init = int(m.group(1)), (m.group(2) or "0")
        data[(N, init, steps)] = (np.array([float(m.group(3)), float(m.group(4))]),
                                  np.array([float(m.group(5)), float(m.group(6))]))
arms = [("0", 800, "scratch", GRAY, "//"), ("S", 800, "denoise PT", BLUE, "//"), ("G", 800, "GFN1 PT", GOLD, "//"),
        ("0", 2300, "scratch", GRAY, ""), ("S", 2300, "denoise PT", BLUE, ""), ("G", 2300, "GFN1 PT", GOLD, "")]
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for k, (ax, lab) in enumerate(zip(axs, ("force MAE (meV/Å)", "energy MAE (meV)"))):
    q = 1 if k == 0 else 0
    for g, N in enumerate((50, 200)):
        for a, (init, st, name, col, hatch) in enumerate(arms):
            x = g * 7.5 + a + (0.5 if a >= 3 else 0)
            v = data[(N, init, st)][q]
            ax.bar(x, v.mean(), 0.85, color=col, hatch=hatch, edgecolor="white" if hatch else col,
                   label=(f"{name}, {st} FT steps") if (g == 0 and k == 0) else None)
            ax.plot([x, x], v, "o", color=NAVY, ms=5)
    ax.set_xticks([2.75, 10.25]); ax.set_xticklabels(["N = 50", "N = 200"])
    ax.set_ylabel(lab)
    panel_label(ax, "ab"[k], x=-0.12)
axs[0].set_ylim(0, 260); axs[1].set_ylim(0, 95)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=13, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(w_pad=3, rect=(0, 0, 1, 0.86))
fig.savefig("../../ch07/figures/Fig7.4.1_pretraining_ethanol.png"); plt.close(fig)
print({k: (v[0].mean().round(1), v[1].mean().round(1)) for k, v in data.items()})
