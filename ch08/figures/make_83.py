"""Figures for Sec.8.3: Fig8.3.1 (band-gap MLP: loss slices, sharpness vs OOD, input gradient vs OOD),
Fig8.3.2 (SchNet-type MLIP: force MAE vs temperature and lambda_max).  Reads the lab outputs in ../labs."""
import os, sys, json
import numpy as np, torch, matplotlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common')); from bookstyle import *
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); LABS = os.path.join(HERE, "..", "labs")
src = open(os.path.join(LABS, "83_bandgap_sam.py")).read().split("sub = np.random.default_rng(1)")[0]
src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))', f'HERE = {LABS!r}')
g = {"__file__": os.path.join(LABS, "83_bandgap_sam.py")}; exec(compile(src, "83", "exec"), g)
Z, Y, make_mlp, S = g["Z"], g["Y"], g["make_mlp"], g["S"]
sp = np.load(os.path.join(LABS, "83_bandgap_split.npz")); tr_all = sp["tr_all"]
R = json.load(open(os.path.join(LABS, "83_bandgap_results.json")))["res"]
IR = json.load(open(os.path.join(LABS, "83_input_robust_results.json")))
TAGS = [("AdamW (bs 64)", "adamw", NAVY, "o", "-"), ("AdamW (bs 1024)", "adamw_bs", GRAY, "s", "--"),
        ("SAM", "sam", RED, "^", "-"), ("ASAM", "asam", BLUE, "v", "-."), ("F-SAM", "fsam", PURPLE, "D", ":"),
        ("AdamW + input noise", "adamw_noise", GREEN, "P", "--"), ("AdamW + FGSM", "adamw_fgsm", GOLD, "X", "-.")]

Xs, ys = Z(tr_all), Y(tr_all)
alphas = np.linspace(-1.0, 1.0, 41)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
slices = {}
for name, tag, c, mk, ls in TAGS[:4]:
    curves = []
    for dseed in range(3):
        m = make_mlp(Xs.shape[1]); m.load_state_dict(torch.load(os.path.join(LABS, f"83_bg_{tag}_s0.pt")))
        d = S.filter_normalized_direction(m, seed=dseed)
        curves.append(S.loss_slice(m, lambda: (m(Xs).squeeze(-1) - ys).pow(2).mean(), d, alphas))
    cm = np.mean(curves, 0); slices[name] = cm.tolist()
    ax[0].plot(alphas, cm, color=c, ls=ls, lw=2.4, label=name)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"step along filter-normalised direction $\alpha$")
ax[0].set_ylabel(r"training MSE (eV$^2$)"); ax[0].legend(frameon=False, fontsize=13, loc="upper center")
panel_label(ax[0], "a")
for name, tag, c, mk, ls in TAGS:
    lam = [r["lam"] for r in R[name]]; ox = [r["ox"] for r in R[name]]
    ax[1].scatter(lam, ox, color=c, marker=mk, s=70, label=name, edgecolor="k", lw=0.4)
    gn = IR["rows"][name]["gnorm"]
    ax[2].scatter(gn, ox, color=c, marker=mk, s=70, edgecolor="k", lw=0.4)
ax[1].set_xlabel(r"$\lambda_{\max}$ of training Hessian"); ax[1].set_ylabel("oxide (OOD) MAE (eV)")
ax[1].axhline(1.622, color=GRAY, ls=":", lw=1.5); ax[1].text(45, 1.64, "predict train mean", color=GRAY, fontsize=13, ha="right")
h_, l_ = ax[1].get_legend_handles_labels()
panel_label(ax[1], "b")
ax[2].set_xlabel(r"mean input-gradient norm $\|\nabla_x f\|$"); ax[2].set_ylabel("oxide (OOD) MAE (eV)")
ax[2].axhline(1.622, color=GRAY, ls=":", lw=1.5)
panel_label(ax[2], "c")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=14, bbox_to_anchor=(0.62, -0.01))
fig.savefig(os.path.join(HERE, "Fig8.3.1_sam_bandgap.png")); plt.close(fig)
json.dump(slices, open(os.path.join(LABS, "83_slices.json"), "w"))
print("slice: MSE at alpha=+/-0.5 (mean of both sides) / alpha=0:")
for k, v in slices.items():
    v = np.array(v); print("  %-16s %.4f / %.4f  ratio %.1f" % (k, 0.5 * (v[10] + v[30]), v[20], 0.5 * (v[10] + v[30]) / v[20]))

# ---- Fig 8.3.2: MLIP -------------------------------------------------------------------------------------
M = json.load(open(os.path.join(LABS, "83_mlip_results.json")))["res"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
T = [300, 600, 1000]; off = {"Adam": -12, "SAM": 0, "ASAM": 12}
sty = {"Adam": (NAVY, "o", "-"), "SAM": (RED, "^", "--"), "ASAM": (BLUE, "v", "-.")}
for k, rows in M.items():
    c, mk, ls = sty[k]
    F = 1e3 * np.array([[r["F300"], r["F600"], r["F1000"]] for r in rows])
    for f in F: ax[0].plot(np.array(T) + off[k], f, color=c, marker=mk, ls="none", alpha=0.45, ms=7)
    ax[0].plot(np.array(T) + off[k], F.mean(0), color=c, marker=mk, ls=ls, lw=2.2, ms=9, label=k)
    lam = [r["lam"] for r in rows]
    ax[1].scatter(lam, F[:, 2], color=c, marker=mk, s=90, edgecolor="k", lw=0.4, label=k)
ax[0].set_xlabel("test temperature (K)"); ax[0].set_ylabel(r"force MAE (meV/$\AA$)"); ax[0].set_xticks(T)
ax[0].legend(frameon=False); panel_label(ax[0], "a")
ax[1].set_xscale("log"); ax[1].set_xticks([300, 500, 1000]); ax[1].set_xticklabels(["300", "500", "1000"])
ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_xlabel(r"$\lambda_{\max}$ of training loss"); ax[1].set_ylabel(r"1000 K force MAE (meV/$\AA$)")
panel_label(ax[1], "b")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "Fig8.3.2_sam_mlip.png")); plt.close(fig)
print("saved")
