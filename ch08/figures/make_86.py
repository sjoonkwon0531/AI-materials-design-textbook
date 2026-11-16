"""Figures for Sec.8.6 (Fig8.6.1, Fig8.6.2). Run after labs/86_dro.py."""
import os, sys, importlib
import numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); sys.path.insert(0, "../../ch08/labs")
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
torch.set_num_threads(1)
M = importlib.import_module("86_dro")
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = "../../ch08/labs"
z = np.load(os.path.join(LABS, "86_dro_results.npz"))

# ------------------------------------------------------------ Fig 8.6.1 landscapes + training
fig, ax = plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
g = np.linspace(0, 1, 121); XX, YY = np.meshgrid(g, g)
st = torch.load(os.path.join(LABS, "86_dro.pt")); net = M.DRO(); net.load_state_dict(st["bsf"]); net.eval()
for k, (K, c, seed) in enumerate([(6, 0.3, 3), (12, 0.15, 3)]):
    G = M.sample_gmm(1, np.random.default_rng(seed), K=K, c=c)
    # slice through the best grid point along x3
    g3 = np.linspace(0, 1, 31); P3 = np.stack(np.meshgrid(g3, g3, g3, indexing="ij"), -1).reshape(-1, 3)
    x3 = P3[np.argmax(M.f_np(G, P3[None]))][2]
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, x3)], -1)[None]
    F = M.f_np(G, pts)[0].reshape(XX.shape)
    cs = ax[k].contourf(XX, YY, F, levels=14, cmap="Blues")
    fig.colorbar(cs, ax=ax[k], fraction=0.046, pad=0.03)
    with torch.no_grad():
        Y = M.run_dro(net, G, np.array([[0.5, 0.5, 0.5]]), M.f_torch, gen=torch.Generator().manual_seed(0))
    # recover the queried x by re-running with recording
    x = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32); state = None; xs = [x.numpy()[0]]
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for t in range(M.HORIZON):
            y = M.f_torch(G, x); mean, L, state = net.step(x, y, state)
            x = (mean + (L @ torch.randn(x.shape, generator=gen)[..., None])[..., 0]).clamp(0, 1); xs.append(x.numpy()[0])
    xs = np.array(xs)
    ax[k].scatter(xs[:, 0], xs[:, 1], s=22, color=GOLD, edgecolor="k", lw=0.5, zorder=3,
                  label="DRO-bsf queries (projected)")
    ax[k].set_xlabel("$x_1$"); ax[k].set_ylabel("$x_2$"); ax[k].set_aspect("equal")
    panel_label(ax[k], "ab"[k], x=-0.2)
pass  # legend described in the caption
lo, lb = z["log_oi"], z["log_bsf"]
ax[2].plot(lo[:, 0], lo[:, 2], color=RED, marker="s", ls="--", lw=2, label="loss 'oi' (authors' default)")
ax[2].plot(lb[:, 0], lb[:, 2], color=NAVY, marker="o", lw=2, label="loss 'bsf' (best so far)")
ax[2].set_xlabel("training iteration (batch 128)"); ax[2].set_ylabel("mean best-of-50 value")
ax[2].set_ylim(0.5, 1.0); ax[2].legend(frameon=False, fontsize=14, loc="lower right")
panel_label(ax[2], "c")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.6.1_dro_landscapes.png")); plt.close(fig)

# ------------------------------------------------------------ Fig 8.6.2 regret curves
labels = ["in-distribution: K=6, c=0.3", "OOD-rugged: K=12, c=0.15", "OOD-noisy: K=6, c=0.3, sigma_y=0.05"]
style = {"DRO-oi (stochastic)": (RED, "--", None), "DRO-bsf (stochastic)": (NAVY, "-", None),
         "DRO-bsf (deterministic)": (NAVY, ":", None), "Nelder-Mead": (PURPLE, "-.", None),
         "CMA-ES": (GREEN, "-", None), "Random search": (GRAY, "--", None), "GP-EI": (GOLD, "-", None)}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.0), sharey=True)
for k, lab in enumerate(labels):
    for name, (col, ls, _) in style.items():
        r = z[f"{lab}__{name}"].mean(0)
        ax[k].semilogy(np.arange(len(r)), r, color=col, ls=ls, lw=2.3, label=name)
    ax[k].set_xlabel("number of experiments"); ax[k].set_ylim(1e-3, 1)
    panel_label(ax[k], "abc"[k])
ax[0].set_ylabel("mean regret (best so far)")
ax[2].legend(frameon=False, fontsize=12, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.6.2_dro_regret.png")); plt.close(fig)
print("done")
