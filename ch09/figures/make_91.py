"""Figures for Section 9.1: Fig9.1.1 (map of the four ways) and Fig9.1.2 (Neural ODE vs HNN, from labs/91_hnn.py)."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
from bookstyle import *
from matplotlib.patches import FancyBboxPatch

OUT = "../../ch09/figures/"

# ------------------------------------------------------------------ Fig 9.1.1
def fig_map():
    cols = ["Physics\n(equation / right-hand side)", "Discretization\n(mesh, basis)", "Solver\n(integrator, linear solve)", "Training signal", "Amortized over"]
    rows = [
        ("Classical solver\n(Ch.2)", ["given", "mesh / stencil", "integrator + Lax", "none", "nothing"], "kkkxx"),
        ("PINN\n(Sec. 9.2)", ["given, in the loss", "network ansatz", "optimizer", "PDE residual", "nothing"], "kLLxx"),
        ("Neural operator\n(Sec. 9.3)", ["implicit in data", "network", "network", "solver data", "one PDE family"], "xLLxx"),
        ("Neural ODE / HNN\n(Sec. 9.1)", ["learned f or H", "state variables", "integrator kept", "trajectories", "initial states"], "Lkkxx"),
        ("Hybrid correction", ["given + learned closure", "coarse mesh kept", "integrator kept", "fine-grid data", "one PDE family"], "Mkkxx"),
        ("Prior-fitted /\nfoundation (Sec. 9.4)", ["implicit in data", "network", "network", "many solvers' data", "many PDEs"], "xLLxx"),
    ]
    colw = [3.3, 2.5, 3.1, 2.4, 2.2]; x0 = 2.9; rowh = 1.0
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    xs = [x0 + sum(colw[:j]) for j in range(len(colw))]
    style = {"k": ("#E3E8F4", NAVY), "L": ("#F6D5D0", RED), "M": ("#FBE6B8", "#8A5A00"), "x": ("white", GRAY)}
    for j, c in enumerate(cols):
        ax.text(xs[j] + colw[j] / 2, len(rows) * rowh + 0.45, c, ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)
    for i, (name, cells, codes) in enumerate(rows):
        y = (len(rows) - 1 - i) * rowh
        ax.text(x0 - 0.15, y + rowh / 2, name, ha="right", va="center", fontsize=16, fontweight="bold")
        for j, (txt, code) in enumerate(zip(cells, codes)):
            fc, ec = style[code]
            ax.add_patch(FancyBboxPatch((xs[j] + 0.06, y + 0.08), colw[j] - 0.12, rowh - 0.16, boxstyle="round,pad=0.01,rounding_size=0.08",
                                        fc=fc, ec=ec if code != "x" else "#BBBBBB", lw=1.4))
            ax.text(xs[j] + colw[j] / 2, y + rowh / 2, txt, ha="center", va="center", fontsize=15, color=ec if code != "x" else "black")
    # legend
    ly = -0.75
    for k, (lab, code) in enumerate((("kept from classical numerics", "k"), ("replaced by a network", "L"), ("partly learned", "M"))):
        fc, ec = style[code]; lx = x0 + 0.2 + k * 4.4
        ax.add_patch(FancyBboxPatch((lx, ly), 0.5, 0.36, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
        ax.text(lx + 0.65, ly + 0.18, lab, va="center", fontsize=15)
    ax.set_xlim(0, x0 + sum(colw) + 0.1); ax.set_ylim(-0.95, len(rows) * rowh + 0.95); ax.axis("off")
    fig.savefig(OUT + "Fig9.1.1_map.png"); plt.close(fig)


# ------------------------------------------------------------------ Fig 9.1.2
def fig_hnn():
    d = json.load(open("../../ch09/data/91_hnn.json"))
    t = np.array(d["t"]); ref = np.array(d["ref"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(wspace=0.33, width_ratios=[1, 1.15, 1.15]))
    ax = axs[0]
    tr = np.array(d["train_noisy"])
    ax.plot(tr[:, :, 0].ravel(), tr[:, :, 1].ravel(), ".", color="#C8C8C8", ms=3, zorder=0, label="training snapshots")
    mlp = np.array(d["NeuralODE-MLP"]["traj"]); hnn = np.array(d["HNN"]["traj"])
    ax.plot(mlp[:, 0], mlp[:, 1], color=RED, lw=0.9, label="neural ODE (MLP)")
    ax.plot(hnn[:, 0], hnn[:, 1], color=BLUE, lw=1.6, label="HNN")
    ax.plot(ref[:420, 0], ref[:420, 1], "--", color=NAVY, lw=1.6, label="exact orbit")
    ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.2, 3.2)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=13, columnspacing=0.8, handlelength=1.5)
    panel_label(ax, "a")
    ax = axs[1]
    E0 = 1 - np.cos(2.0)
    ax.plot(t, d["NeuralODE-MLP"]["E"], color=RED, lw=1.8, label="neural ODE (MLP)")
    ax.plot(t, d["HNN"]["E"], color=BLUE, lw=1.8, ls="-", label="HNN")
    ax.axhline(E0, color=NAVY, ls="--", lw=1.4, label="exact")
    ax.set_xlabel("time"); ax.set_ylabel("true energy H(q, p)"); ax.set_ylim(0.4, 1.6)
    ax.legend(frameon=False, loc="lower left"); panel_label(ax, "b")
    ax = axs[2]
    m = (t >= 160)
    ax.plot(t[m], ref[m, 0], "--", color=NAVY, lw=1.6, label="exact")
    ax.plot(t[m], mlp[m, 0], color=RED, lw=1.6, label="neural ODE (MLP)")
    ax.plot(t[m], hnn[m, 0], color=BLUE, lw=1.6, label="HNN")
    ax.set_xlabel("time"); ax.set_ylabel("q(t)"); ax.set_ylim(-2.4, 3.6)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=13, columnspacing=0.8, handlelength=1.5); panel_label(ax, "c")
    fig.savefig(OUT + "Fig9.1.2_hnn.png"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["map", "hnn"]
    if "map" in which: fig_map()
    if "hnn" in which: fig_hnn()
