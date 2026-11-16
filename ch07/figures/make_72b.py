"""Figures for Sec. 7.2.3 (owner B): Fig7.2.4 graph construction, Fig7.2.6 SchNet vs ACSF-BPNN,
Fig7.2.5 distance-only limit (CH4).  Usage: python make_72b.py [4|5|6 ...]"""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
FIG = "../ch07/figures"
LAB = "../ch07/labs"
which = sys.argv[1:] or ["4", "5", "6"]
COL = {6: GRAY, 8: RED, 1: BLUE}


def fig4():
    d = np.load("../ch07/data/ethanol_xtb_300K.npz")
    X, Z = d["R"][0], d["Z"]
    X = X - X.mean(0)
    # project on the plane of the heavy atoms
    u, s, vt = np.linalg.svd(X[:3] - X[:3].mean(0))
    P = X @ vt[:2].T
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.05, 1.1, 0.9]))
    ax = axs[0]
    for i in range(9):
        for j in range(i + 1, 9):
            if D[i, j] < 1.7:
                ax.plot(*P[[i, j]].T, color=NAVY, lw=3, zorder=1)
            elif D[i, j] < 2.5:
                ax.plot(*P[[i, j]].T, color=GOLD, lw=1.6, ls="--", zorder=0)
    for i in range(9):
        ax.add_patch(Circle(P[i], 0.17 if Z[i] > 1 else 0.11, color=COL[Z[i]], zorder=3))
    ax.plot([], [], color=NAVY, lw=3, label=r"$r_c$ = 1.7 Å (16 edges)")
    ax.plot([], [], color=GOLD, lw=1.6, ls="--", label=r"added at $r_c$ = 2.5 Å (48)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=14)
    ax.set_aspect("equal"); ax.axis("off"); panel_label(ax, "a", x=0.0)
    # (b) honeycomb crystal (2D analogue of diamond Si): 2-atom cell and images
    ax = axs[1]
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
    A0 = np.array([0.0, 0.0]); B0 = (a1 + a2) / 3
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            T = n1 * a1 + n2 * a2
            for p, c in [(A0, NAVY), (B0, RED)]:
                q = p + T
                if np.abs(q).max() < 1.6:
                    ax.add_patch(Circle(q, 0.07, color=c, alpha=1.0 if (n1, n2) == (0, 0) else 0.35, zorder=3))
    cell = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
    ax.plot(*cell.T, color=GRAY, lw=1.5)
    rc = 0.7
    ax.add_patch(Circle(A0, rc, fill=False, ls=":", color=NAVY, lw=1.8))
    neigh = [B0, B0 - a1, B0 - a2]
    labs = ["(0,0)", "(−1,0)", "(0,−1)"]
    for q, l in zip(neigh, labs):
        ax.add_patch(FancyArrowPatch(q, A0, arrowstyle="-|>", mutation_scale=16, color=RED, lw=2, zorder=2,
                                     shrinkA=6, shrinkB=6))
        off = {"(0,0)": (0.10, 0.06), "(−1,0)": (-0.62, 0.08), "(0,−1)": (0.12, -0.24)}[l]
        ax.text(q[0] + off[0], q[1] + off[1], l, fontsize=14, color=RED)
    ax.text(0.03, 0.62, r"$r_c$", fontsize=16, color=NAVY)
    ax.set_xlim(-1.3, 1.6); ax.set_ylim(-1.1, 1.3); ax.set_aspect("equal"); ax.axis("off")
    panel_label(ax, "b", x=0.0)
    # (c) quotient multigraph A <- B with three parallel edges
    ax = axs[2]
    pa, pb = np.array([0.0, 0.0]), np.array([2.0, 0.0])
    for rad, l in zip([0.45, 0.0, -0.45], labs):
        ax.add_patch(FancyArrowPatch(pb, pa, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=18, color=RED, lw=2, shrinkA=22, shrinkB=22))
        ax.text(1.0, {0.45: 0.52, 0.0: 0.07, -0.45: -0.66}[rad], l, ha="center", fontsize=14, color=RED)
    ax.add_patch(Circle(pa, 0.22, color=NAVY)); ax.add_patch(Circle(pb, 0.22, color=RED))
    ax.text(*pa, "A", color="white", ha="center", va="center", fontsize=17, fontweight="bold")
    ax.text(*pb, "B", color="white", ha="center", va="center", fontsize=17, fontweight="bold")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-1.0, 1.0); ax.set_aspect("equal"); ax.axis("off")
    panel_label(ax, "c", x=0.0)
    fig.savefig(f"{FIG}/Fig7.2.4_graph_construction.png"); plt.close(fig)


def fig5():
    res = json.load(open(f"{LAB}/72b_mlip_results.json"))
    Ns = np.array([100, 300, 1000])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    for k, (name, c, mk, ls) in enumerate([("BPNN(ACSF)", GOLD, "s", "--"), ("SchNet", NAVY, "o", "-")]):
        e = np.array([res[f"{name}|{n}"][0] for n in Ns]); f = np.array([res[f"{name}|{n}"][1] for n in Ns])
        lab = "ACSF + MLP (BPNN)" if k == 0 else "SchNet-like GNN"
        axs[0].semilogx(Ns, e, marker=mk, ls=ls, color=c, ms=9, lw=2, label=lab)
        axs[1].semilogx(Ns, f, marker=mk, ls=ls, color=c, ms=9, lw=2, label=lab)
    axs[0].set_xlabel("training frames $N$"); axs[0].set_ylabel("energy MAE (meV)")
    axs[1].set_xlabel("training frames $N$"); axs[1].set_ylabel("force MAE (meV/Å)")
    for ax in axs[:2]:
        ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.minorticks_off(); ax.set_ylim(bottom=0)
    axs[1].legend(frameon=False)
    p = np.load(f"{LAB}/72b_parity_schnet.npz")
    sel = np.random.default_rng(0).choice(p["F"].size, 4000, replace=False)
    axs[2].scatter(p["F"].ravel()[sel], p["Fp"].ravel()[sel], s=5, color=NAVY, alpha=0.4, rasterized=True)
    lim = [-6, 6]; axs[2].plot(lim, lim, color=RED, lw=1.5, ls="--")
    axs[2].set_xlim(lim); axs[2].set_ylim(lim); axs[2].set_aspect("equal")
    axs[2].set_xlabel("xTB force (eV/Å)"); axs[2].set_ylabel("SchNet force (eV/Å)")
    for ax, l in zip(axs, "abc"):
        panel_label(ax, l)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig7.2.6_schnet_vs_acsf.png"); plt.close(fig)


def fig6():
    r = 1.09
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3) * r
    sq = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]) * r
    fig = plt.figure(figsize=(16, 5.2))
    from numpy.polynomial.legendre import legval
    for k, (X, l) in enumerate([(tet, "a"), (sq, "b")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for h in X:
            ax.plot([0, h[0]], [0, h[1]], [0, h[2]], color=NAVY, lw=3)
        for i in range(4):
            for j in range(i + 1, 4):
                dd = np.linalg.norm(X[i] - X[j])
                ax.plot(*np.array([X[i], X[j]]).T, color=GOLD if dd < 2.0 else GRAY, ls="--", lw=1.5)
        ax.scatter([0], [0], [0], s=420, color=GRAY, depthshade=False)
        ax.scatter(*X.T, s=200, color=BLUE, depthshade=False)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(18, 30 if k == 0 else 25)
        lim = 1.0; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.text2D(0.02, 0.95, l, transform=ax.transAxes, fontsize=24, fontweight="bold")
    ax = fig.add_subplot(1, 3, 3)
    ls_ = np.arange(1, 7)
    for X, lab, c, mk in [(tet, "tetrahedral", NAVY, "o"), (sq, "square planar", RED, "s")]:
        u = X / np.linalg.norm(X, axis=1, keepdims=True)
        cth = (u @ u.T)[~np.eye(4, dtype=bool)]
        S = [legval(cth, [0] * l + [1]).sum() for l in ls_]
        ax.plot(ls_, S, marker=mk, color=c, ms=10, lw=2, ls="-" if c == NAVY else "--", label=lab)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("angular order $l$"); ax.set_ylabel(r"$\sum_{j\neq k} P_l(\cos\theta_{jk})$")
    ax.legend(frameon=False, loc="upper left"); ax.set_xticks(ls_)
    panel_label(ax, "c")
    fig.savefig(f"{FIG}/Fig7.2.5_distance_only_limit.png"); plt.close(fig)


for w in which:
    {"4": fig4, "5": fig5, "6": fig6}[w]()
    print("done", w)
