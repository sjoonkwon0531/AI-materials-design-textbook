"""Fig 7.5.1: LoRA vs full fine-tuning vs linear probe vs adapters (labs/75_lora_results.json, 75_dW_svals.npy)."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import matplotlib.pyplot as plt
LAB = "../ch07/labs/"
res = json.load(open(LAB + "75_lora_results.json")); sv = np.load(LAB + "75_dW_svals.npy")
fig, axs = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
mk = {"linear probe": ("v", GREEN), "full": ("s", RED), "adapter": ("D", PURPLE), "scratch": ("x", GRAY)}
ax = axs[0]
for N, ls in ((50, "--"), (200, "-")):
    r = [res[f"{N}|LoRA r={k}"] for k in (1, 2, 4, 8)]
    ax.errorbar([x["params"] for x in r], [np.mean(x["tgt"]) for x in r], [np.std(x["tgt"]) for x in r],
                fmt="o" + ls, color=NAVY, ms=7, capsize=3, label="LoRA r=1,2,4,8" if N == 200 else None,
                mfc="white" if N == 50 else NAVY)
    for meth, (m, c) in mk.items():
        x = res[f"{N}|{meth}"]
        ax.errorbar(x["params"], np.mean(x["tgt"]), np.std(x["tgt"]), fmt=m, color=c, ms=9, capsize=3,
                    mfc="white" if N == 50 else c, label=meth if N == 200 else None)

z = np.mean(res["50|zero-shot"]["tgt"])
ax.axhline(z, color=GOLD, lw=2, ls=":", label="zero-shot")
ax.set_xscale("log"); ax.set_xlabel("trainable parameters"); ax.set_ylabel("target-test MAE (eV)")
ax.set_ylim(0.5, 0.9); ax.set_xlim(80, 4e5)
ax.legend(frameon=False, fontsize=12.5, loc="upper left", ncol=2)
ax.text(0.03, 0.04, "open: N = 50   filled: N = 200", transform=ax.transAxes, fontsize=14, color=GRAY)
panel_label(ax, "a", x=-0.16)
ax = axs[1]
offs = {"full": (-34, 10), "linear probe": (6, 4), "adapter": (6, 6), "LoRA r=1": (-70, -18), "LoRA r=4": (8, -14), "scratch": (-50, -4)}
for meth, m, c in (("full", "s", RED), ("linear probe", "v", GREEN), ("adapter", "D", PURPLE),
                   ("LoRA r=1", "o", NAVY), ("LoRA r=4", "o", BLUE), ("scratch", "x", GRAY)):
    x = res[f"200|{meth}"]
    ax.plot(np.mean(x["tgt"]), np.mean(x["src"]), m, color=c, ms=10)
    ax.annotate(meth, (np.mean(x["tgt"]), np.mean(x["src"])), xytext=offs[meth], textcoords="offset points", fontsize=13)
x = res["200|zero-shot"]
ax.plot(np.mean(x["tgt"]), np.mean(x["src"]), "*", color=GOLD, ms=16)
ax.annotate("pretrained", (np.mean(x["tgt"]), np.mean(x["src"])), xytext=(6, -14), textcoords="offset points", fontsize=13)
ax.set_xlabel("target-test MAE (eV)"); ax.set_ylabel("source-test MAE (eV)")
ax.set_xlim(0.53, 0.70); ax.set_ylim(0.37, 0.88)
panel_label(ax, "b", x=-0.2)
ax = axs[2]
k = np.arange(1, len(sv) + 1)
ax.semilogy(k, sv, "-", color=NAVY, lw=2)
ax.semilogy(k[:8], sv[:8], "o", color=RED, ms=6)
ax.set_xlabel("singular value index"); ax.set_ylabel(r"$\sigma_k(\Delta W)$")
ax.set_xlim(0, 260)
panel_label(ax, "c", x=-0.2)
fig.tight_layout(w_pad=2)
fig.savefig("../ch07/figures/Fig7.5.1_lora_finetuning.png"); plt.close(fig)
print("done")
