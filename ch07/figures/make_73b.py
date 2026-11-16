"""Figures 7.3.5-7.3.7 for Sec. 7.3.4-7.3.6 (owner C). Numbers come from e3nn directly or from labs/73b_*.npy/out."""
import sys, re, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from e3nn import o3
torch.set_default_dtype(torch.float64)
OUT = "../../ch07/figures/"
LAB = "../../ch07/labs/"

# ---------------------------------------------------------------- Fig 7.3.5 real spherical harmonics + equivariance error
fig = plt.figure(figsize=(13, 6.6))
gs = gridspec.GridSpec(4, 9, figure=fig, width_ratios=[1]*7 + [0.35, 3.2], wspace=0.0, hspace=0.0)
th, ph = np.meshgrid(np.linspace(0, np.pi, 60), np.linspace(0, 2*np.pi, 120), indexing="ij")
xyz = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], -1)
pts = torch.tensor(xyz.reshape(-1, 3))
first = None
for l in range(4):
    Y = o3.spherical_harmonics(l, pts, normalize=True, normalization="integral").numpy()
    for k, m in enumerate(range(-l, l + 1)):
        ax = fig.add_subplot(gs[l, 3 - l + k], projection="3d")
        if first is None: first = ax
        v = Y[:, k].reshape(th.shape)
        r = 0.55 * np.abs(v) / np.abs(Y).max()
        col = np.where(v[..., None] >= 0, np.array(matplotlib.colors.to_rgba(BLUE)), np.array(matplotlib.colors.to_rgba(RED)))
        ax.plot_surface(r*xyz[..., 0], r*xyz[..., 1], r*xyz[..., 2], facecolors=col, linewidth=0, antialiased=True, shade=True)
        ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_zlim(-0.7, 0.7)
        ax.set_box_aspect((1, 1, 1)); ax.axis("off"); ax.view_init(20, 35)
        ax.text2D(0.5, -0.06, f"$k={k}$", transform=ax.transAxes, ha="center", fontsize=12, color=GRAY)
    axl = fig.add_subplot(gs[l, 0]) if l < 3 else None
    fig.text(0.035, 0.80 - l*0.205, f"$l={l}$", fontsize=18, ha="left", va="center")
    if axl is not None: axl.axis("off")
fig.text(0.02, 0.95, "a", fontsize=24, fontweight="bold")
# panel b: equivariance error from the lab output
txt = open(LAB + "73b_sh_wigner.out").read()
e32 = [float(x) for x in re.search(r"float32\s+max error by l=0..6: (.*)", txt).group(1).split()]
e64 = [float(x) for x in re.search(r"float64\s+max error by l=0..6: (.*)", txt).group(1).split()]
axb = fig.add_subplot(gs[:, 8])
ls = np.arange(1, 7)
axb.semilogy(ls, e32[1:], "o-", color=NAVY, label="float32", ms=8)
axb.semilogy(ls, e64[1:], "s--", color=RED, label="float64", ms=8)
axb.set_xlabel("degree $l$"); axb.set_ylabel(r"max $|Y_l(Rr) - D^l(R)Y_l(r)|$")
axb.set_ylim(1e-16, 1e-3); axb.legend(frameon=False, loc="center right")
panel_label(axb, "b", x=-0.30)
fig.savefig(OUT + "Fig7.3.5_spherical_harmonics.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7.3.6 tensor product: block diagonalisation, nonlinearity, parameters
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
R = o3.rand_matrix()
D1 = o3.Irrep(1, -1).D_from_matrix(R)
U = torch.cat([(2*L+1)**0.5 * o3.wigner_3j(1, 1, L).reshape(9, 2*L+1) for L in (0, 1, 2)], 1)   # 9 x 9 orthogonal
M = U.T @ torch.kron(D1, D1) @ U
ax = axs[0]
im = ax.imshow(np.abs(M.numpy()), cmap="Blues", vmin=0, vmax=1)
for b in (0.5, 3.5):
    ax.axhline(b, color=GRAY, lw=0.8); ax.axvline(b, color=GRAY, lw=0.8)
ax.set_xticks([0, 2, 6]); ax.set_xticklabels(["$L=0$", "$L=1$", "$L=2$"])
ax.set_yticks([0, 2, 6]); ax.set_yticklabels(["$L=0$", "$L=1$", "$L=2$"])
ax.spines[:].set_visible(True)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$|U^{T}(D^1\otimes D^1)U|$")
panel_label(ax, "a", x=-0.28)
txt = open(LAB + "73b_tp_layer.out").read()
f32 = float(re.search(r"float32.*?\n\s+max equivariance error: SO\(3\) ([0-9.e+-]+)", txt).group(1))
f64 = float(re.search(r"float64.*?\n\s+max equivariance error: SO\(3\) ([0-9.e+-]+)", txt).group(1))
naive = float(re.search(r"component-wise SiLU on vectors: max error = ([0-9.]+)", txt).group(1))
gate = float(re.search(r"norm gate sigma\(\|v\|\) v:\s+max error = ([0-9.e+-]+)", txt).group(1))
ax = axs[1]
labels = ["layer\nfp32", "layer\nfp64", "norm\ngate", "SiLU\ncomp."]
vals = [f32, f64, gate, naive]
cols = [NAVY, BLUE, GREEN, RED]
ax.bar(range(4), vals, color=cols, width=0.65)
ax.set_yscale("log"); ax.set_ylim(1e-17, 1e2)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=14)
ax.set_ylabel("max equivariance error")
for i, v in enumerate(vals): ax.text(i, v*3, f"{v:.0e}" if v < 1 else f"{v:.1f}", ha="center", fontsize=13)
panel_label(ax, "b", x=-0.22)
ax = axs[2]
ntp = int(re.search(r"TP weights per edge \(from radial MLP\) = (\d+)", txt).group(1))
nbil = int(re.search(r"unconstrained bilinear map .*? has (\d+) entries", txt).group(1))
nlin_e = int(re.search(r"\(dim 144\): (\d+) weights", txt).group(1))
nlin = int(re.search(r"torch.nn.Linear 144->144: (\d+) params", txt).group(1))
x = np.array([0, 1]); w = 0.36
ax.bar(x - w/2, [ntp, nlin_e], w, color=NAVY, label="equivariant")
ax.bar(x + w/2, [nbil, nlin], w, color=GRAY, hatch="//", edgecolor="white", label="unconstrained")
ax.set_yscale("log"); ax.set_ylim(1e2, 2e5)
ax.set_xticks(x); ax.set_xticklabels(["tensor product\n(per edge)", "linear mixing\n(dim 144)"], fontsize=14)
ax.set_ylabel("number of weights")
for xi, (a, b) in zip(x, [(ntp, nbil), (nlin_e, nlin)]):
    ax.text(xi - w/2, a*1.3, f"{a:,}", ha="center", fontsize=13); ax.text(xi + w/2, b*1.3, f"{b:,}", ha="center", fontsize=13)
ax.legend(frameon=False, loc="upper left", fontsize=14)
panel_label(ax, "c", x=-0.22)
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT + "Fig7.3.6_tensor_product.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7.3.7 angular resolution vs cost
txt = open(LAB + "73b_bodyorder_cost.out").read()
line = re.search(r"l_max=3: \|A_l\| \(l=0..3\) tetra = ([0-9. ]+)\| square = ([0-9. ]+)\|", txt)
tet = [float(v) for v in line.group(1).split()]; sq = [float(v) for v in line.group(2).split()]
rows = np.load(LAB + "73b_cost_lmax.npy")
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]; l = np.arange(4); w = 0.38
ax.bar(l - w/2, tet, w, color=NAVY, label="tetrahedral CH$_4$")
ax.bar(l + w/2, sq, w, color=GOLD, hatch="//", edgecolor="white", label="square-planar CH$_4$")
ax.set_xticks(l); ax.set_xlabel("degree $l$ of the centre feature")
ax.set_ylabel(r"invariant $|A_l|$ (random layer)")
ax.set_ylim(0, 0.42); ax.legend(frameon=False, loc="upper right")
panel_label(ax, "a")
ax = axs[1]
ax.semilogy(rows[:, 0], rows[:, 4]*1e3, "o-", color=NAVY, ms=8, label="fwd+bwd time (ms)")
ax.semilogy(rows[:, 0], rows[:, 3], "s--", color=RED, ms=8, label="TP weights per edge")
ax.semilogy(rows[:, 0], rows[:, 2], "^:", color=GREEN, ms=8, label="TP paths")
ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel(r"$l_{\max}$")
ax.set_ylabel("cost of one convolution")
ax.set_ylim(0.5, 1e6); ax.legend(frameon=False, loc="upper left", fontsize=14)
panel_label(ax, "b")
fig.tight_layout(w_pad=3)
fig.savefig(OUT + "Fig7.3.7_lmax_resolution_cost.png"); plt.close(fig)
print("done")
