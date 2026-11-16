"""Figures for Section 9.2 (data written by labs/92_*.py into ../data/)."""
import sys, json, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
from bookstyle import *
from matplotlib.colors import LogNorm

OUT = "../ch09/figures/"; DATA = "../ch09/data/"
load = lambda n: json.load(open(DATA + n))
PI = math.pi
u_exact = lambda x, t, D=0.1: np.exp(-D * PI**2 * t) * np.sin(PI * x) + 0.5 * np.exp(-9 * D * PI**2 * t) * np.sin(3 * PI * x)


def fig_heat():
    d = load("92_heat.json"); A = d["A"]
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.1]))
    for ax, key, lab in ((axs[0], "soft_err", "a"), (axs[1], "hard_err", "b")):
        E = np.array(A[key]); E = np.maximum(E, 1e-7)
        im = ax.pcolormesh(np.linspace(0, 1, 101), np.linspace(0, 1, 101), E.T, norm=LogNorm(1e-6, 5e-3), cmap="magma", shading="auto", rasterized=True)
        ax.set_xlabel("x"); ax.set_ylabel("t"); panel_label(ax, lab, x=-0.2)
        cb = fig.colorbar(im, ax=ax, pad=0.03); cb.set_label("|error|")
    ax = axs[2]
    cn = np.array(d["CN"])
    ax.loglog(cn[:, 4], cn[:, 2], "s-", color=NAVY, lw=2, ms=8, label="Crank-Nicolson")
    for i, (N, e, w) in enumerate(zip(cn[:, 0], cn[:, 2], cn[:, 4])):
        if N in (25, 100, 400): ax.annotate(f"N={int(N)}", (w, e), textcoords="offset points", xytext=(8, 4), fontsize=13, color=NAVY)
    ax.loglog(A["soft"]["wall"], A["soft"]["rel"], "o", color=RED, ms=10, label="PINN, soft constraints")
    ax.loglog(A["hard"]["wall"], A["hard"]["rel"], "^", color=BLUE, ms=11, label="PINN, hard constraints")
    ax.set_xlabel("CPU time (s)"); ax.set_ylabel("relative L2 error"); ax.set_ylim(1e-6, 1e-1)
    ax.legend(frameon=False, loc="upper left", fontsize=14); panel_label(ax, "c", x=-0.2)
    fig.savefig(OUT + "Fig9.2.1_heat_forward.png"); plt.close(fig)


def fig_inverse():
    d = load("92_inverse.json")["B"]
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]
    x = np.linspace(0, 1, 101); t = np.linspace(0, 1, 101); X, T = np.meshgrid(x, t, indexing="ij")
    cs = ax.contourf(X, T, u_exact(X, T), levels=14, cmap="YlGnBu")
    xd, td, _ = d["B1_data"]; ax.plot(xd, td, "o", color=RED, ms=6, mec="white", mew=0.6, label="B1: 40 scattered points")
    xd, td, _ = d["B2_data"]; ax.plot(xd, td, "s", color=NAVY, ms=5, mec="white", mew=0.5, label="B2: 5 sensors x 16 times")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, fontsize=12.5, handletextpad=0.2, columnspacing=0.8)
    fig.colorbar(cs, ax=ax, pad=0.03).set_label("u(x, t)"); panel_label(ax, "a", x=-0.2)
    ax = axs[1]
    for key, c, ls, lab in (("B1_traj", RED, "-", "B1 (IC, BC known)"), ("B2_traj", NAVY, "--", "B2 (IC, BC unknown)")):
        tr = np.array(d[key]); ax.semilogy(tr[:, 0], tr[:, 1], color=c, ls=ls, lw=2.2, label=lab)
    ax.axhline(0.1, color=GRAY, ls=":", lw=1.8); ax.text(50, 0.107, "true D = 0.1", color=GRAY, fontsize=14)
    ax.axvline(3000, color=GOLD, lw=1.5); ax.text(3040, 0.6, "L-BFGS", color="#8A5A00", fontsize=14)
    ax.set_xlabel("iteration"); ax.set_ylabel("estimated D"); ax.set_ylim(0.05, 1.0); ax.legend(frameon=False, loc="lower left"); panel_label(ax, "b", x=-0.2)
    ax = axs[2]
    groups = [("B1\nPINN", d["B1"]["D"], RED, "o"), ("B1\nCN + LSQ", d["B1_classical"], GRAY, "s"), ("B2\nPINN", d["B2"]["D"], NAVY, "o"), ("B2\nsine LSQ", d["B2_classical"], GRAY, "s")]
    for i, (lab, vals, c, m) in enumerate(groups):
        ax.plot(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, m, color=c, ms=9, alpha=0.9)
        ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=c, lw=2.5)
    ax.axhline(0.1, color=GRAY, ls=":", lw=1.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=14); ax.set_ylabel("estimated D"); ax.set_xlim(-0.5, 3.5)
    panel_label(ax, "c", x=-0.2)
    fig.savefig(OUT + "Fig9.2.2_heat_inverse.png"); plt.close(fig)


def fig_ad():
    d = load("92_ad.json")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]; fd = np.array(d["fd"])
    ax.loglog(fd[:, 0], fd[:, 1], "o-", color=NAVY, lw=2, ms=7, label="finite difference, float64")
    ax.loglog(fd[:, 0], fd[:, 2], "s--", color=RED, lw=2, ms=7, label="finite difference, float32")
    ax.axhline(d["ad64"], color=NAVY, ls=":", lw=2); ax.text(1.3e-6, d["ad64"] * 2.5, "autodiff, float64", color=NAVY, fontsize=14)
    ax.axhline(d["ad32"], color=RED, ls=":", lw=2); ax.text(1.3e-6, d["ad32"] * 2.5, "autodiff, float32", color=RED, fontsize=14)
    ax.set_xlabel("step size h"); ax.set_ylabel("max relative error of $u_{xx}$"); ax.set_ylim(1e-17, 1e5)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=14); panel_label(ax, "a", x=-0.18)
    ax = axs[1]; c = np.array(d["cost"]); k = c[:, 0]
    ax.semilogy(k, c[:, 1] / d["t_fwd"], "o-", color=NAVY, lw=2, ms=8, label="nested reverse mode")
    ax.semilogy(k, c[:, 3] / d["t_fwd"], "s--", color=RED, lw=2, ms=8, label="nested forward mode (jvp)")
    ax.semilogy(k, c[:, 2] / d["t_fwd"], "^:", color=BLUE, lw=2, ms=8, label="reverse + parameter gradient")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("derivative order k"); ax.set_ylabel("cost / forward pass")
    ax.legend(frameon=False, loc="upper left", fontsize=14); panel_label(ax, "b", x=-0.18)
    fig.savefig(OUT + "Fig9.2.3_autodiff.png"); plt.close(fig)


def fig_pdes():
    A = load("92_pdes_A.json"); B = load("92_pdes_B.json")["B"]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6), gridspec_kw=dict(wspace=0.3, hspace=0.36))
    ax = axs[0, 0]; x = np.array(A["A_pinn"]["x"]); U = np.array(A["A_pinn"]["u"]); R = np.array(A["A_pinn"]["ref"])
    cols = [NAVY, BLUE, GREEN, GOLD, RED]
    for j, tt in enumerate((0, 0.25, 0.5, 0.75, 1.0)):
        ax.plot(x, R[:, j], color=cols[j], lw=3.2, alpha=0.35)
        ax.plot(x, U[:, j], "--", color=cols[j], lw=1.6, label=f"t = {tt:g}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x, t)"); ax.legend(frameon=False, fontsize=13, loc="upper right"); panel_label(ax, "a")
    ax = axs[0, 1]
    fd = np.array(A["A_fd"]); ax.loglog(fd[:, 2], fd[:, 1], "s-", color=NAVY, lw=2, ms=8, label="finite difference + LSODA")
    for N, e, w in fd:
        if N in (100, 400, 1600): ax.annotate(f"N={int(N)}", (w, e), textcoords="offset points", xytext=(8, 4), fontsize=13, color=NAVY)
    ax.loglog(A["A_wall"], A["A_errs"], "o", color=RED, ms=10, label="PINN (2 seeds)")
    ax.set_xlabel("CPU time (s)"); ax.set_ylabel("relative L2 error"); ax.set_ylim(5e-5, 1.0); ax.legend(frameon=False, loc="upper center", fontsize=14); panel_label(ax, "b")
    ax = axs[1, 0]; x = np.array(B["x"]); C = np.array(B["c"]); R = np.array(B["ref"])
    for j, tt in enumerate((0, 0.25, 0.5, 0.75, 1.0)):
        ax.plot(x, R[:, j], color=cols[j], lw=3.2, alpha=0.35)
        ax.plot(x, C[:, j], "--", color=cols[j], lw=1.6, label=f"t = {tt:g}")
    ax.set_xlabel("x"); ax.set_ylabel("c(x, t)"); ax.set_ylim(-1.15, 1.6); ax.legend(frameon=False, fontsize=13, ncol=3, loc="upper center", columnspacing=0.8); panel_label(ax, "c")
    ax = axs[1, 1]; t = np.linspace(0, 1, 101)
    ax.plot(t, B["Fref"], color=NAVY, lw=3.2, alpha=0.35, label="free energy, spectral")
    ax.plot(t, B["F"], "--", color=NAVY, lw=1.8, label="free energy, PINN")
    ax.set_xlabel("t"); ax.set_ylabel("free energy F[c]")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.semilogy(t[1:], np.maximum(np.array(B["mass"])[1:], 1e-18), color=RED, lw=1.8, ls="-.", label="mass drift, PINN")
    ax2.set_ylabel("|mean c(t) - mean c(0)|", color=RED); ax2.tick_params(axis="y", colors=RED); ax2.set_ylim(1e-7, 1e-1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=13.5); panel_label(ax, "d")
    fig.savefig(OUT + "Fig9.2.4_burgers_ch.png"); plt.close(fig)


def fig_spectral():
    d = load("92_spectral.json")
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw=dict(wspace=0.33))
    ax = axs[0]; x = np.linspace(0, 1, 2001)[::2]; ex = np.sin(2 * PI * x) + 0.1 * np.sin(50 * PI * x)
    ax.plot(x, ex, color=GRAY, lw=3.5, alpha=0.45, label="exact")
    ax.plot(x, d["B"]["FF sigma=10, Adam, equal weights"]["pred"], color=RED, lw=1.6, ls="-.", label="Adam, equal weights")
    ax.plot(x, d["B"]["FF sigma=10, Adam, NTK weights"]["pred"], color=GREEN, lw=1.6, ls=":", label="Adam, NTK weights")
    ax.plot(x, d["B"]["FF sigma=10, Levenberg-Marquardt"]["pred"], color=NAVY, lw=1.3, ls="--", label="Levenberg-Marquardt")
    ax.set_xlim(0, 0.5); ax.set_xlabel("x"); ax.set_ylabel("u(x)"); ax.set_ylim(-0.75, 2.5)
    ax.legend(frameon=False, fontsize=12.5, loc="upper left", bbox_to_anchor=(0.01, 1.02), ncol=2, columnspacing=0.8, handlelength=1.6); panel_label(ax, "a", x=-0.17)
    ax = axs[1]
    sty = {"tanh MLP, Adam": (GRAY, "-", "MLP, Adam"), "FF sigma=10, Adam, equal weights": (RED, "-.", "FF, Adam, equal"),
           "FF sigma=10, Adam, NTK weights": (GREEN, ":", "FF, Adam, NTK"), "FF sigma=10, Adam + L-BFGS": (GOLD, "-", "FF, Adam + L-BFGS"),
           "FF sigma=10, Levenberg-Marquardt": (NAVY, "--", "FF, LM"), "tanh MLP, Levenberg-Marquardt": (PURPLE, "-.", "MLP, LM")}
    for name, (c, ls, lab) in sty.items():
        h = np.array(d["B"][name]["hist"]); ax.loglog(np.maximum(h[:, 0], 0.05), h[:, 1], color=c, ls=ls, lw=2.2, label=lab)
    ax.set_xlabel("CPU time (s)"); ax.set_ylabel("relative L2 error"); ax.set_ylim(1e-8, 3e3)
    ax.legend(frameon=False, fontsize=12.5, loc="lower left", ncol=2, columnspacing=0.6, handlelength=1.8); panel_label(ax, "b", x=-0.17)
    ax = axs[2]
    for (name, c, ls) in (("tanh MLP", GRAY, "-"), ("FF sigma=1", GREEN, ":"), ("FF sigma=10", NAVY, "--"), ("FF sigma=50", GOLD, "-.")):
        e = np.sort(np.array(d["C"][name]["eig_rr"]))[::-1]; e = np.maximum(e, 1e-12)
        ax.loglog(np.arange(1, len(e) + 1), e, color=c, ls=ls, lw=2.2, label=name.replace("sigma=", ""))
        ax.plot([1], [d["C"][name]["lam_uu"]], "o", color=c, ms=8, mfc="white", mew=2)
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("NTK eigenvalue"); ax.set_ylim(1e-8, 1e14)
    ax.legend(frameon=False, fontsize=13.5, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, columnspacing=0.8, handlelength=1.5, handletextpad=0.4); panel_label(ax, "c", x=-0.17)
    fig.savefig(OUT + "Fig9.2.5_spectral_ntk.png"); plt.close(fig)


def fig_convection(beta=30):
    d = load("92_convection.json")
    fig = plt.figure(figsize=(17.5, 9.2)); gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.4)
    xs = np.linspace(0, 2 * PI, 256, endpoint=False)[::4]; ts = np.linspace(0, 1, 101)[::2]
    X, T = np.meshgrid(xs, ts, indexing="ij")
    panels = [("exact", np.sin(X - beta * T)), ("vanilla", np.array(d[f"vanilla_{beta}"]["u"])), ("curriculum", np.array(d["curriculum"]["u"]))]
    for j, (name, U) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        im = ax.pcolormesh(ts, xs, U, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
        ax.set_xlabel("t"); ax.set_ylabel("x"); panel_label(ax, "abc"[j], x=-0.14)
        ax.text(0.03, 0.93, name, transform=ax.transAxes, fontsize=15, va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
        fig.colorbar(im, ax=ax, pad=0.02)
    ax = fig.add_subplot(gs[1, 0]); tt = np.linspace(0, 1, 101)
    for key, c, ls, lab in ((f"vanilla_{beta}", RED, "-", "vanilla"), ("curriculum_adam", GRAY, "-", "curriculum, Adam stages"), ("curriculum", NAVY, "--", "curriculum, L-BFGS stages"), ("marching", GREEN, "-.", "time marching"), ("periodic", GOLD, ":", "periodic embedding")):
        ax.semilogy(tt[1:], np.array(d[key]["err_t"])[1:], color=c, ls=ls, lw=2.2, label=lab)
    ax.set_xlabel("t"); ax.set_ylabel("relative L2 error at time t"); ax.set_ylim(2e-7, 5); ax.legend(frameon=False, fontsize=13.5, loc="lower right", handlelength=1.8); panel_label(ax, "d", x=-0.14)
    ax = fig.add_subplot(gs[1, 1])
    betas = [b for b in (1, 10, 20, 30, 40, 50) if f"vanilla_{b}" in d]
    for b in betas:
        e = d[f"vanilla_{b}"]["errs"]; ax.semilogy([b] * len(e), e, "o", color=RED, ms=10)
    ax.semilogy([beta + 1.2] * 2, d["curriculum"]["errs"], "s", color=NAVY, ms=9, label="curriculum, L-BFGS stages")
    ax.semilogy([beta + 2.4] * 2, d["marching"]["errs"], "^", color=GREEN, ms=10, label="time marching")
    ax.semilogy([beta - 1.2] * 2, d["curriculum_adam"]["errs"], "v", color=GRAY, ms=9, label="curriculum, Adam stages")
    ax.semilogy([beta - 2.4] * 2, d["periodic"]["errs"], "D", color=GOLD, ms=9, label="periodic embedding")
    ax.plot([], [], "o", color=RED, ms=10, label="vanilla")
    ax.set_xlabel(r"convection speed $\beta$"); ax.set_ylabel("relative L2 error"); ax.set_ylim(1e-4, 5); ax.legend(frameon=False, fontsize=13.5, loc="upper left"); panel_label(ax, "e", x=-0.14)
    ax = fig.add_subplot(gs[1, 2])
    labs_, vals = [], []
    for b in betas:
        Lr, Li, Lb = d[f"vanilla_{b}"]["loss"]; labs_.append(str(b)); vals.append((Lr, Li, Lb))
    vals = np.array(vals); w = 0.26; xx = np.arange(len(betas))
    for k, (c, lab) in enumerate(((NAVY, "PDE residual"), (RED, "initial condition"), (GOLD, "periodic BC"))):
        ax.bar(xx + (k - 1) * w, np.maximum(vals[:, k], 1e-9), w, color=c, label=lab)
    ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(labs_); ax.set_xlabel(r"convection speed $\beta$"); ax.set_ylabel("final loss term (vanilla)")
    ax.legend(frameon=False, fontsize=13.5, loc="upper left"); panel_label(ax, "f", x=-0.14)
    fig.savefig(OUT + "Fig9.2.6_convection.png"); plt.close(fig)


if __name__ == "__main__":
    fs = dict(heat=fig_heat, inverse=fig_inverse, ad=fig_ad, pdes=fig_pdes, spectral=fig_spectral, convection=fig_convection)
    for k in (sys.argv[1:] or fs):
        fs[k]()
