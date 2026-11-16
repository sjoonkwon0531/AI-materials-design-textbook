"""Figures for Section 6.4. Reads labs/64_*_results.npz written by the lab scripts."""
import sys, os, logging
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.join(HERE, "..", "labs")
CMAP = LinearSegmentedColormap.from_list("gap", [BLUE, GOLD, RED])

def gapscatter(ax, E, y, s=5):
    m = y == 0
    ax.scatter(E[m, 0], E[m, 1], s=s, color="#BBBBBB", alpha=0.6, lw=0, label="metal (gap 0)")
    sc = ax.scatter(E[~m, 0], E[~m, 1], s=s, c=np.clip(y[~m], 0, 6), cmap=CMAP, vmin=0, vmax=6, lw=0, alpha=0.85)
    return sc

# ---------- Fig 6.4.1: PCA ----------
P = np.load(os.path.join(LAB, "64_pca_results.npz"), allow_pickle=True)
fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
ax = axs[0]; k = np.arange(1, 41)
ax.plot(k, np.cumsum(P["evr"])[:40], color=NAVY, lw=2.4, marker="o", ms=4, label="standardized")
ax.plot(k, np.cumsum(P["evr_raw"])[:40], color=RED, lw=2.4, ls="--", marker="s", ms=4, label="raw units")
ax.axhline(0.9, color=GRAY, lw=1.2, ls=":")
ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative explained variance")
ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="lower right")
panel_label(ax, "a")
ax = axs[1]
sc = gapscatter(ax, P["T"][:, :2], P["y"], s=6)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Band gap (eV)")
ax.legend(frameon=False, loc="lower right", markerscale=3, fontsize=13)
panel_label(ax, "b")
ax = axs[2]
L = P["load"][0]; names = P["names"]; o = np.argsort(-np.abs(L))[:10][::-1]
ax.barh(np.arange(10), L[o], color=[RED if v < 0 else NAVY for v in L[o]])
ax.set_yticks(np.arange(10)); ax.set_yticklabels(names[o], fontsize=12.5)
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("PC1 loading")
panel_label(ax, "c", x=-0.75)
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.93, wspace=0.25)
pos = axs[2].get_position(); axs[2].set_position([pos.x0 + 0.1, pos.y0, pos.width - 0.1, pos.height]); fig.savefig(os.path.join(HERE, "Fig6.4.1_pca.png")); plt.close(fig)

# ---------- Fig 6.4.2: t-SNE / UMAP ----------
E = np.load(os.path.join(LAB, "64_embed_results.npz"), allow_pickle=True)
y = E["y"]
keys = ["t-SNE_(perp_5)", "t-SNE_(perp_30)", "t-SNE_(perp_100)", "UMAP_(k=5)", "UMAP_(k=15)", "UMAP_(k=15,_seed_1)"]
fig, axs = plt.subplots(2, 3, figsize=(17, 11))
for ax, kk, lab in zip(axs.ravel(), keys, "abcdef"):
    sc = gapscatter(ax, E[kk], y, s=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["left", "bottom"]: ax.spines[sp].set_visible(False)
    txt = kk.replace("_", " ").replace("perp", "perplexity").replace("k=", "n_neighbors = ").replace(", seed 1", ", seed 1")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=14, color=GRAY)
    panel_label(ax, lab, x=-0.02)
fig.subplots_adjust(right=0.9, wspace=0.08, hspace=0.12)
cax = fig.add_axes([0.92, 0.3, 0.015, 0.4]); cb = fig.colorbar(sc, cax=cax); cb.set_label("Band gap (eV)")
fig.savefig(os.path.join(HERE, "Fig6.4.2_tsne_umap.png")); plt.close(fig)

# ---------- Fig 6.4.3: synthetic blobs ----------
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
S, lab = E["syn_S"], E["syn_lab"]
embs = [PCA(2).fit_transform(S), TSNE(2, perplexity=30, init="pca", random_state=0, n_jobs=2).fit_transform(S),
        umap.UMAP(n_neighbors=15, random_state=0).fit_transform(S)]
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for ax, Em, pl in zip(axs, embs, "abc"):
    for i, (c, mk, nm) in enumerate(zip([NAVY, RED, GREEN], ["o", "^", "s"], ["A (s.d. 1)", "B (s.d. 4)", "C (s.d. 1)"])):
        ax.scatter(Em[lab == i, 0], Em[lab == i, 1], s=10, color=c, marker=mk, alpha=0.6, label=nm)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim")
    panel_label(ax, pl, x=-0.02)
axs[0].legend(frameon=False, loc="upper center", markerscale=2, fontsize=13, ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.4.3_blobs.png")); plt.close(fig)

# ---------- Fig 6.4.4: spectral unmixing ----------
U = np.load(os.path.join(LAB, "64_unmix_results.npz"))
tt = U["tt"]; cols = [NAVY, RED, GREEN]; lss = ["-", "--", "-."]
def norm(s): return s / np.abs(s).max()
fig, axs = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
ax = axs[0, 0]
for i in range(3):
    ax.plot(tt, norm(U["S2"][i]) + 1.2 * (2 - i), color=cols[i], lw=1.8, ls=lss[i], label=f"phase {i+1}")
for j, idx in enumerate([0, 1, 2]):
    ax.plot(tt, norm(U["X2"][idx]) * 0.9 - 1.3 - 1.0 * j, color=GRAY, lw=1.2)
ax.set_yticks([]); ax.set_ylabel("Intensity (offset)")
ax.legend(frameon=False, loc="upper right", fontsize=13, ncol=3, bbox_to_anchor=(1.0, 1.12))
panel_label(ax, "a", x=-0.08)
for ax, key, pl in [(axs[0, 1], "S2_pca", "b"), (axs[1, 0], "S2_ica", "c"), (axs[1, 1], "S2_nmf", "d")]:
    for i in range(3):
        off = 1.3 * (2 - i)
        ax.plot(tt, norm(U["S2"][i]) + off, color="#CCCCCC", lw=3.2)
        ax.plot(tt, norm(U[key][i]) + off, color=cols[i], lw=1.4, ls=lss[i])
    ax.set_yticks([]); panel_label(ax, pl, x=-0.08)
for ax in axs[1]: ax.set_xlabel(r"2$\theta$ (deg)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig6.4.4_unmixing.png")); plt.close(fig)
print("done")
