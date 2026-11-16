"""Figures for Sec.8.4: Fig8.4.1 (domain randomisation heat maps + trajectories), Fig8.4.2 (BC vs DAgger)."""
import os, sys, json, importlib.util
import numpy as np, matplotlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); from bookstyle import *
from matplotlib.colors import LogNorm
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = os.path.join(HERE, "..", "labs")
_s = importlib.util.spec_from_file_location("env84", os.path.join(LABS, "84_thermal_env.py"))
E = importlib.util.module_from_spec(_s); _s.loader.exec_module(E)
SCALE = np.array([1, 1, 0.1, 1, 10, 1, 1, 1.0])
pol = lambda th: (lambda o: np.hstack([o, np.ones((len(o), 1))]) * SCALE @ np.asarray(th))
D = json.load(open(os.path.join(LABS, "84_domain_rand_results.json")))
H_GRID = [0.05, 0.07, 0.1, 0.14, 0.2]; D_GRID = [1, 2, 3, 4, 5]

fig = plt.figure(figsize=(17, 10.5)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.42, wspace=0.9)
names = [("nominal", "a"), ("DR", "b"), ("DR-CVaR(25%)", "c")]
norm = LogNorm(5, 400)
for k, (nm, lab) in enumerate(names):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
    G = 1e3 * np.mean([r["grid"] for r in D[nm]], 0)
    im = ax.imshow(G, origin="lower", cmap="viridis", norm=norm, aspect="auto")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "%.0f" % G[i, j], ha="center", va="center", fontsize=12.5, color="w" if G[i, j] < 60 else "k")
    ax.set_xticks(range(5)); ax.set_xticklabels(D_GRID); ax.set_yticks(range(5)); ax.set_yticklabels(H_GRID)
    ax.set_xlabel("dead time d (s)"); ax.set_ylabel(r"heat loss h (s$^{-1}$)")
    ax.add_patch(plt.Rectangle((0.5, 1.5), 1, 1, fill=False, ec=RED, lw=2.5))                  # nominal cell
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 4, 5, fill=False, ec="w", lw=1.5, ls="--"))     # DR training range
    ax.spines[:].set_visible(False); panel_label(ax, lab, x=-0.25)
cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.015, pad=0.01); cb.set_label("mean |error| (K)")
for k, (h, d, lab) in enumerate([(0.1, 2, "d"), (0.05, 4, "e")]):
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
    t = np.arange(1, E.T_EP + 1)
    ax.plot(t, 1e3 * E.target(t), color="k", lw=1.5, ls=":", label="target")
    for nm, c, ls in (("nominal", NAVY, "-"), ("DR", RED, "--"), ("DR-CVaR(25%)", GREEN, "-.")):
        x = E.rollout(pol(D[nm][0]["theta"]), h, 0.2, d, 1, np.random.default_rng(5), record=True)["x"][0]
        ax.plot(t, 1e3 * x, color=c, ls=ls, lw=2.2, label=nm)
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$T-T_{\mathrm{amb}}$ (K)")
    ax.text(0.98, 0.05, f"h = {h}, d = {d}", transform=ax.transAxes, ha="right", fontsize=15)
    if k == 0: ax.legend(frameon=False, fontsize=13, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    panel_label(ax, lab)
fig.savefig(os.path.join(HERE, "Fig8.4.1_domain_rand.png")); plt.close(fig)

# ---- Fig 8.4.2 --------------------------------------------------------------------------------------------
A = json.load(open(os.path.join(LABS, "84_dagger_results.json")))
src = open(os.path.join(LABS, "84_dagger.py")).read().split("exp_cost =")[0].replace(
    "HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {LABS!r}")
g = {}; exec(compile(src, "84_dagger", "exec"), g)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
n = A["runs"][0]["n"]
for key, c, mk, ls, lab in (("bc", NAVY, "o", "-", "behaviour cloning"), ("dagger", RED, "^", "--", "DAgger")):
    Y = 1e3 * np.array([r[key] for r in A["runs"]])
    for y_ in Y: ax[0].plot(n, y_, color=c, ls="none", marker=mk, ms=5, alpha=0.35)
    ax[0].plot(n, np.median(Y, 0), color=c, marker=mk, ls=ls, lw=2.2, ms=9, label=lab + " (median of 5)")
ax[0].axhline(1e3 * A["expert"], color=GRAY, ls=":", lw=1.8); ax[0].text(n[-1], 1e3 * A["expert"] * 1.12, "expert", color=GRAY, ha="right", fontsize=14)
ax[0].set_yscale("log"); ax[0].set_xlabel("expert-labelled samples"); ax[0].set_ylabel("mean |error| (K)")
ax[0].legend(frameon=False); panel_label(ax[0], "a")
Ts = sorted(int(k) for k in A["horizon"])
for key, c, mk, ls, lab in (("bc", NAVY, "o", "-", "BC"), ("dagger", RED, "^", "--", "DAgger")):
    ex = np.array([Ts[i] * (np.array(A["horizon"][str(T)][key]) - A["horizon"][str(T)]["expert"]) for i, T in enumerate(Ts)])
    ax[1].fill_between(Ts, ex.min(1), ex.max(1), color=c, alpha=0.15)
    ax[1].plot(Ts, ex.mean(1), color=c, marker=mk, ls=ls, lw=2.2, ms=9, label=lab + " (mean, range of 5)")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks(Ts); ax[1].set_xticklabels(Ts)
ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_xlabel("episode length T (s)"); ax[1].set_ylabel("excess total cost over expert (K s)")
ax[1].legend(frameon=False); panel_label(ax[1], "b")
r = g["run"](0, return_models=True); pb, pd = r["models"]
t = np.arange(1, E.T_EP + 1)
ax[2].plot(t, 1e3 * E.target(t), color="k", lw=1.5, ls=":", label="target")
for nm, p, c, ls in (("expert", g["expert"], GRAY, "-"), ("BC", pb.predict, NAVY, "-"), ("DAgger", pd.predict, RED, "--")):
    X = E.rollout(p, *g["P"], 5, np.random.default_rng(11), record=True)["x"]
    for i, x in enumerate(X): ax[2].plot(t, 1e3 * x, color=c, ls=ls, lw=1.6, alpha=0.85, label=nm if i == 0 else None)
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel(r"$T-T_{\mathrm{amb}}$ (K)"); ax[2].legend(frameon=False, fontsize=13, loc="lower right")
panel_label(ax[2], "c")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.4.2_dagger.png")); plt.close(fig)
print("saved")
