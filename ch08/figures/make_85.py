"""Figures for Sec.8.5 (Fig8.5.1-8.5.3). Run after labs/85_ppo.py and labs/85_sam_surrogate.py."""
import os, sys, importlib
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); sys.path.insert(0, "../../ch08/labs")
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY
E = importlib.import_module("85_film_env")
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = "../../ch08/labs"
P = E.FilmParams()

# ------------------------------------------------------------ Fig 8.5.1 environment physics
def basis(P):
    x = np.arange(P.L); L = P.L; B = []
    for m in range(1, L // 2): B.append(np.sqrt(2 / L) * np.cos(2 * np.pi * m * x / L))
    for m in range(1, L // 2): B.append(np.sqrt(2 / L) * np.sin(2 * np.pi * m * x / L))
    B.append(np.sqrt(1 / L) * np.cos(np.pi * x)); return np.array(B)
B = basis(P)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, (T, col, ls) in enumerate([(640, BLUE, "-"), (695, NAVY, "-"), (760, RED, "--")]):
    rng = np.random.default_rng(4); c = np.zeros(len(P.m))
    for t in range(P.N): c = E.ou_step(c, T, P, rng)
    h = c @ B
    ax[0].plot(np.arange(P.L), h + 8 * (2 - k), color=col, ls=ls, lw=1.8,
               label=f"T = {T} K, w = {np.sqrt((c ** 2).sum() / P.L):.2f}")
ax[0].set_xlabel("lateral position x (sites)"); ax[0].set_ylabel("height - mean (ML), offset")
ax[0].legend(frameon=False, loc="upper center", fontsize=13, ncol=3, handlelength=1.5, columnspacing=1.0)
ax[0].set_ylim(-9, 40)
panel_label(ax[0], "a")
Ts = np.arange(620, 781, 5); W, TBv = [], []
for T in Ts:
    J, w, TB = E.rollout_batch(lambda t, o: np.full(len(o), (T - 700) / 100), 600, 1, P, delta=0.0)
    W.append(w.mean()); TBv.append(P.lam * TB.mean())
W, TBv = np.array(W), np.array(TBv)
ax[1].semilogy(Ts, W, color=BLUE, marker="o", ms=5, label="final roughness $w_N$")
ax[1].semilogy(Ts, TBv, color=RED, ls="--", marker="s", ms=5, label=r"$\lambda\,\times$ thermal budget")
ax[1].semilogy(Ts, W + TBv, color=NAVY, lw=2.6, label=r"cost $-J$")
i = int(np.argmin(W + TBv)); ax[1].axvline(Ts[i], color=GRAY, ls=":", lw=1.5)
ax[1].set_xlabel("constant substrate temperature (K)"); ax[1].set_ylabel("cost (units of w)")
ax[1].set_ylim(1e-3, 3e3); ax[1].legend(frameon=False, loc="upper left", fontsize=14)
panel_label(ax[1], "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.5.1_film_env.png")); plt.close(fig)
print("Fig8.5.1: optimum constant actual T", Ts[i], "cost", (W + TBv)[i])

# ------------------------------------------------------------ Fig 8.5.2 PPO
z = np.load(os.path.join(LABS, "85_ppo_results.npz"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for s, ls in zip(range(3), ["-", "--", ":"]):
    c = z[f"curve_{s}"]; ax[0].plot(c[:, 0] / 1e3, c[:, 1], color=NAVY, ls=ls, lw=2, label="PPO (closed loop)" if s == 0 else None)
    c = z[f"curve_ol{s}"]; ax[0].plot(c[:, 0] / 1e3, c[:, 1], color=GOLD, ls=ls, lw=2, label="PPO (clock only)" if s == 0 else None)
for key, col, lab in [("tuned two-level", GREEN, "tuned two-level (test)"), ("oracle (knows delta)", RED, "oracle (test)")]:
    ax[0].axhline(z[f"{key}__J"].mean(), color=col, ls="-.", lw=1.6, label=lab)
ax[0].set_xlabel("environment steps (thousands)"); ax[0].set_ylabel("episode return")
ax[0].set_ylim(-3.2, -1.7); ax[0].legend(frameon=False, fontsize=13, loc="lower right")
panel_label(ax[0], "a")
T, d = z["PPO seed 0__T"], z["PPO seed 0__d"]
Tset = (T - d[:, None])[:, 4:].mean(1); idx = np.arange(0, len(d), 4)
ax[1].scatter(d[idx], Tset[idx], s=10, color=NAVY, alpha=0.5, label="PPO, mean set-point (steps 5-12)")
dd = np.linspace(-40, 40, 2)
ax[1].plot(dd, 695 - dd, color=RED, ls="--", lw=2, label="oracle: 695 K - offset")
ax[1].plot(dd, [685, 685], color=GREEN, ls="-.", lw=2, label="tuned constant 685 K")
ax[1].set_xlabel("hidden temperature offset (K)"); ax[1].set_ylabel("set-point (K)")
ax[1].legend(frameon=False, fontsize=13, loc="upper right"); ax[1].set_ylim(630, 770)
panel_label(ax[1], "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.5.2_ppo.png")); plt.close(fig)

# ------------------------------------------------------------ Fig 8.5.3 SAM surrogate
r = np.load(os.path.join(LABS, "85_sam_surrogate_results.npz"))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw=dict(width_ratios=[1, 1.35]))
mk = {"AdamW": ("o", NAVY), "SAM rho=0.05": ("s", BLUE), "SAM rho=0.2": ("D", GREEN), "ASAM rho=0.5": ("^", PURPLE)}
for m, (mm, col) in mk.items():
    ax[0].scatter(r[f"lam__{m}"], r[f"{m}__mae_id"], marker=mm, color=col, s=60,
                  label=m.replace("rho", r"$\rho$"))
ax[0].set_xlabel(r"$\lambda_{\max}$ of training-loss Hessian"); ax[0].set_ylabel("in-distribution MAE of J")
ax[0].legend(frameon=False, fontsize=13); panel_label(ax[0], "a", y=1.18)
methods = ["AdamW", "SAM rho=0.05", "SAM rho=0.2", "ASAM rho=0.5", "Ensemble-5 LCB", "AdamW + trust region",
           "SAM rho=0.2 + trust region", "Ensemble LCB + trust region"]
short = ["AdamW", "SAM .05", "SAM .2", "ASAM", "Ens-LCB", "AdamW\n+TR", "SAM .2\n+TR", "Ens-LCB\n+TR"]
xs = np.arange(len(methods))
for j, m in enumerate(methods):
    pr, tr = r[f"{m}__pred"], r[f"{m}__true"]
    ax[1].scatter(np.full(5, j) - 0.12, pr, marker="o", facecolor="none", edgecolor=BLUE, s=45,
                  label="surrogate prediction" if j == 0 else None)
    ax[1].scatter(np.full(5, j) + 0.12, tr, marker="o", color=RED, s=45, label="true J (simulator)" if j == 0 else None)
ax[1].axhline(float(r["J_twolevel"]), color=GREEN, ls="-.", lw=1.6, label="tuned two-level")
ax[1].axhline(float(r["J_hist"]), color=GOLD, ls="--", lw=1.6, label="best historical run")
ax[1].set_yscale("symlog", linthresh=2); ax[1].set_xticks(xs); ax[1].set_xticklabels(short, fontsize=12)
ax[1].set_ylabel("J of proposed schedule")
ax[1].legend(frameon=False, fontsize=13, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, scatterpoints=1)
ax[1].set_ylim(-400, 30)
panel_label(ax[1], "b", x=-0.1, y=1.18)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.5.3_sam_surrogate.png")); plt.close(fig)
print("done")
