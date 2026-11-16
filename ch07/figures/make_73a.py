"""Figures for Sec. 7.3.1-7.3.3 (owner B): Fig7.3.1 raw-xyz Transformer vs SchNet under rotation,
Fig7.3.2 group hierarchy, Fig7.3.3 rotate-then-predict vs predict-then-rotate on ethanol,
Fig7.3.4 symmetry/conservation diagnostics of four force models.  Usage: python make_73a.py [1|2|3|4 ...]"""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
FIG = "../../ch07/figures"; LAB = "../../ch07/labs"
which = sys.argv[1:] or ["1", "2", "3", "4"]


def fig1():
    d = json.load(open(f"{LAB}/73a_attn_results.json"))
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6))
    sty = {"Transformer, raw frames": (GRAY, "x", ":"), "MLP(xyz), canonical": (PURPLE, "v", "-."),
           "Transformer, canonical": (RED, "s", "--"), "Transformer, canonical +aug": (GOLD, "^", "-."),
           "SchNet, raw frames": (NAVY, "o", "-")}
    lab = {"Transformer, raw frames": "Transformer, raw MD frames", "MLP(xyz), canonical": "xyz MLP, canonical frame",
           "Transformer, canonical": "Transformer, canonical frame", "Transformer, canonical +aug": "Transformer, canonical + rot. aug.",
           "SchNet, raw frames": "SchNet (invariant), raw MD frames"}
    ax = axs[0]
    for k, h in d["hist"].items():
        h = np.array(h); c, mk, ls = sty[k]
        ax.plot(h[:, 0], h[:, 2], color=c, ls=ls, lw=2.2, marker=mk, ms=7, markevery=5, label=lab[k])
    ax.axhline(712, color=GRAY, lw=1); ax.text(3000, 760, "zero-force baseline", ha="right", color=GRAY, fontsize=14)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("validation force MAE (meV/Å)")
    ax.set_ylim(50, 2000); ax.legend(frameon=False, fontsize=12.5, loc="lower right", bbox_to_anchor=(1.0, 0.19))
    panel_label(ax, "a")
    ax = axs[1]; th = d["thetas"]
    for k in ["Transformer, canonical", "Transformer, canonical +aug", "SchNet, raw frames"]:
        c, mk, ls = sty[k]
        ax.plot(th, d["sweep"][k], marker=mk, color=c, ls=ls, lw=2.2, ms=8, label=lab[k])
    ax.axhline(712, color=GRAY, lw=1)
    ax.set_xlabel("rotation of the test set (deg)"); ax.set_ylabel("test force MAE (meV/Å)"); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yscale("log"); ax.set_ylim(50, 2000); panel_label(ax, "b")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig7.3.1_attention_vs_gnn.png"); plt.close(fig)


SIZES = {}


def box(ax, key, xy, text, c, w=2.3, h=0.85, fs=17):
    SIZES[key] = (xy, w, h)
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.05", fc=c, ec="k", lw=1.0, alpha=0.95))
    ax.text(*xy, text, ha="center", va="center", fontsize=fs, color="white" if c in (NAVY, RED, PURPLE, BLUE, GREEN) else "k")


def arrow(ax, a, b):
    (pa, wa, ha), (pb, wb, hb) = SIZES[a], SIZES[b]
    ax.add_patch(FancyArrowPatch((pa[0], pa[1] - ha / 2 - 0.06), (pb[0], pb[1] + hb / 2 + 0.06), arrowstyle="-|>",
                                 mutation_scale=16, color=GRAY, lw=1.6))


def fig2():
    fig, ax = plt.subplots(figsize=(15, 7.6))
    box(ax, "E3", (5.6, 6.3), "E(3)\n$x\\mapsto Rx+t$, det $R=\\pm1$", NAVY, w=3.6, h=1.1)
    box(ax, "O3", (2.4, 4.4), "O(3)\n$x\\mapsto Rx$", BLUE, w=2.4, h=1.0)
    box(ax, "SE3", (6.4, 4.4), "SE(3)\n$x\\mapsto Rx+t$, det $R=+1$", BLUE, w=3.4, h=1.0)
    box(ax, "T3", (10.0, 4.4), "T(3)\n$x\\mapsto x+t$", BLUE, w=2.4, h=1.0)
    box(ax, "SO3", (5.0, 2.5), "SO(3)\nproper rotations", GREEN, w=2.8, h=1.0)
    box(ax, "Oh", (1.4, 2.5), "$O_h$ (48)", GOLD, w=1.7, h=0.75)
    box(ax, "Td", (0.5, 0.8), "$T_d$ (24)", GOLD, w=1.6, h=0.75)
    box(ax, "D4h", (2.5, 0.8), "$D_{4h}$ (16)", GOLD, w=1.6, h=0.75)
    box(ax, "C3v", (0.5, -0.8), "$C_{3v}$ (6)", GOLD, w=1.6, h=0.75)
    box(ax, "SO2", (5.0, 0.8), "SO(2) $\\cong$ U(1)\nrotations about $z$", GOLD, w=2.8, h=1.0, fs=15)
    box(ax, "Sn", (10.6, 1.6), "$S_n$ ($n!$ elements)\nswap identical atoms", PURPLE, w=3.2, h=1.0, fs=15)
    for a, b in [("E3", "O3"), ("E3", "SE3"), ("E3", "T3"), ("O3", "SO3"), ("SE3", "SO3"), ("O3", "Oh"),
                 ("Oh", "Td"), ("Oh", "D4h"), ("Td", "C3v"), ("SO3", "SO2")]:
        arrow(ax, a, b)
    ax.text(8.6, 6.4, "arrow: contains as a subgroup", fontsize=15, color=GRAY)
    ax.text(10.6, 0.95, "acts on atom labels,\nnot on space", ha="center", fontsize=14, color=PURPLE, va="top")
    ax.set_xlim(-0.6, 12.4); ax.set_ylim(-1.4, 7.0); ax.axis("off")
    fig.savefig(f"{FIG}/Fig7.3.2_group_hierarchy.png"); plt.close(fig)


def fig3():
    import torch
    from importlib.machinery import SourceFileLoader
    torch.set_num_threads(1)
    U = SourceFileLoader("u72b", f"{LAB}/72b_utils.py").load_module()
    src = open(f"{LAB}/73a_attn.py").read().split("R, Z, E, F = U.load(300)")[0]
    ns = {}; exec(src, ns)
    R, Z, E, F = U.load(300); _, _, te = U.split_300K()
    sch = U.SchNetMol(Z, n_feat=32, n_rbf=20, cutoff=5.0, n_inter=3)
    sch.load_state_dict(torch.load(f"{LAB}/73a_schnet_N1000.pt")); sch.e0 = float(E[U.split_300K()[0][:1000]].mean())
    tfm = ns["XYZTransformer"](Z); tfm.load_state_dict(torch.load(f"{LAB}/73a_transformer_canon_N1000.pt")); tfm.e0 = sch.e0
    ref = R[0] - R[0].mean(0)
    M = ns["kabsch"](R[te[0]][:3], ref[:3]) if "kabsch" in ns else None
    X = (R[te[0]] - R[te[0]].mean(0)) @ M.T             # canonical frame = frame the Transformer was trained in
    th = np.radians(75); Q = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Xr = X @ Q.T
    def pred(m, Y):
        _, Fp = U.ef(m, torch.as_tensor(Y[None], dtype=torch.float32)); return Fp[0].detach().numpy()
    col = {6: GRAY, 8: RED, 1: BLUE}
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.8))
    sc = 0.35
    for ax, Y, lab in [(axs[0], X, "a"), (axs[1], Xr, "b"), (axs[2], Xr, "c")]:
        D = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
        for i in range(9):
            for j in range(i + 1, 9):
                if D[i, j] < 1.7: ax.plot(*Y[[i, j], :2].T, color=GRAY, lw=2.5, zorder=1)
        for i in range(9):
            ax.add_patch(Circle(Y[i, :2], 0.16 if Z[i] > 1 else 0.1, color=col[Z[i]], zorder=3))
        ax.set_aspect("equal"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.axis("off")
        panel_label(ax, lab, x=0.0)
    def quiv(ax, Y, Fv, c, ls="-"):
        for i in range(9):
            ax.annotate("", xy=Y[i, :2] + sc * Fv[i, :2], xytext=Y[i, :2],
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, ls=ls, mutation_scale=14), zorder=4)
    Fs, Ft = pred(sch, X), pred(tfm, X)
    quiv(axs[0], X, Fs, NAVY)
    quiv(axs[1], Xr, pred(sch, Xr), NAVY); quiv(axs[1], Xr, Fs @ Q.T, GOLD, ls="--")
    quiv(axs[2], Xr, pred(tfm, Xr), RED); quiv(axs[2], Xr, Ft @ Q.T, GOLD, ls="--")
    axs[1].plot([], [], color=NAVY, lw=2.2, label="SchNet $F(Qx)$"); axs[1].plot([], [], color=GOLD, lw=2.2, ls="--", label="$Q\\,F(x)$")
    axs[2].plot([], [], color=RED, lw=2.2, label="xyz Transformer $F(Qx)$"); axs[2].plot([], [], color=GOLD, lw=2.2, ls="--", label="$Q\\,F(x)$")
    for ax in axs[1:]:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.14), fontsize=14)
    e_s = np.abs(pred(sch, Xr) - Fs @ Q.T).mean(); e_t = np.abs(pred(tfm, Xr) - Ft @ Q.T).mean()
    print(f"Fig7.3.3: frame {te[0]}, rotation 75 deg about the view axis: mean |F(Qx)-QF(x)| SchNet {e_s*1000:.2e} meV/A,"
          f" Transformer {e_t*1000:.0f} meV/A")
    fig.savefig(f"{FIG}/Fig7.3.3_rotate_then_predict.png"); plt.close(fig)


def fig4():
    s = json.load(open(f"{LAB}/73a_equiv_results.json"))
    names = list(s.keys())
    short = ["(a) SchNet E,\nautograd F", "(b) xyz-MLP E,\nautograd F", "(c) xyz-MLP\ndirect F", "(d) SchNet\ndirect equiv. F"]
    metrics = [("F_so3", "rotation equiv. error"), ("asym", "Jacobian asymmetry"), ("torque", "net torque")]
    floor = 1e-17
    fig, ax = plt.subplots(figsize=(15, 5.6))
    w = 0.25; cols = [NAVY, GOLD, RED]; hat = ["", "//", ".."]
    for k, ((key, lab), c, h) in enumerate(zip(metrics, cols, hat)):
        vals = [max(s[n][key], floor) for n in names]
        ax.bar(np.arange(4) + (k - 1) * w, vals, w, color=c, hatch=h, edgecolor="k", lw=0.6, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-17, 1e3); ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=15)
    ax.axhline(2.2e-16, color=GRAY, ls=":", lw=1.2); ax.text(-0.55, 4e-16, "float64 $\\epsilon$", color=GRAY, fontsize=13, ha="left")
    ax.set_ylabel("relative error"); ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig7.3.4_force_symmetry_tests.png"); plt.close(fig)


for w in which:
    {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[w]()
    print("done", w)
