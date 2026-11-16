"""Figures for §9.4 (Fig9.4.1-9.4.4). Reads data written by labs/94a_incontext.py, 94c_fewshot.py, 94d_uq.py.
Usage: python make_94.py [incontext] [fewshot] [uq] [conformal]"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
D = "../ch09/data/"; F = "../ch09/figures/"
which = sys.argv[1:] or ["incontext", "fewshot", "uq", "conformal"]

if "incontext" in which:
    d = np.load(D + "94a_incontext.npz")
    K, res = d["Ks"], d["res"]                              # columns: LS, Bayes, network, |network - Bayes|
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    a = ax[0]; m = K > 0
    a.loglog(K[m], res[m, 0], "s--", color=GRAY, lw=2.2, ms=9, label="per-mode least squares")
    a.loglog(K[m], res[m, 1], "o-", color=NAVY, lw=2.6, ms=10, label="exact Bayes (grid)")
    a.loglog(K[m], res[m, 2], "x", color=RED, ms=13, mew=3, label="prior-fitted network")
    a.axhline(res[0, 1], color=GOLD, ls=":", lw=2.4); a.text(2.6, res[0, 1] * 1.08, "prior only (K = 0)", color="#9A7B1C", fontsize=15, va="bottom")
    a.axvspan(8.6, 40, color=ICE, alpha=0.5, lw=0); a.text(18.5, 2.6, "K > 8: not in training", fontsize=13.5, ha="center", va="top", color=NAVY)
    a.set_xlabel("Context size K (example pairs)"); a.set_ylabel("Relative L2 error of $u(T)$")
    a.set_xticks([1, 2, 4, 8, 16, 32]); a.set_xticklabels(["1", "2", "4", "8", "16", "32"]); a.set_xlim(0.85, 40); a.set_ylim(0.025, 3)
    a.legend(frameon=False, loc="lower left", fontsize=14)
    a = ax[1]
    nu, c = d["G_NU"], d["G_C"]
    p1 = d["post1"]; p8 = d["post8"]
    cs = (c > 0.3) & (c < 1.5)
    a.pcolormesh(c[cs], nu, p1[:, cs] / p1.max(), cmap="Blues", shading="auto", rasterized=True)
    a.contour(c[cs], nu, p8[:, cs] / p8.max(), levels=[0.1, 0.5], colors=[RED], linewidths=2.2)
    a.plot([0.9], [0.06], marker="*", color=GOLD, ms=24, mec="k", mew=1.2, ls="none")
    a.set_yscale("log"); a.set_ylim(0.02, 0.5); a.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5]); a.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"]); a.minorticks_off(); a.set_xlabel("Advection speed $c$"); a.set_ylabel(r"Diffusivity $\nu$")
    a.text(0.34, 0.33, "K = 1 (shading)", color=NAVY, fontsize=15); a.text(0.34, 0.24, "K = 8 (contours)", color=RED, fontsize=15)
    a = ax[2]; kk = np.arange(1, 9)
    a.plot(kk, d["m_true"].real, "-", color="k", lw=2.4, label="true $m_k$")
    a.plot(kk, d["ls1"].real, "s", color=GRAY, ms=9, mfc="none", mew=2, label="least squares")
    a.plot(kk, d["mb1"].real, "o", color=NAVY, ms=11, mfc="none", mew=2.4, label="exact Bayes")
    a.plot(kk, d["mn1"].real, "x", color=RED, ms=11, mew=3, label="network")
    a.axhline(0, color=GRAY, lw=0.8); a.set_ylim(-2.1, 3.2)
    a.set_xlabel("Wavenumber $k$"); a.set_ylabel(r"Re $R(k)$ inferred from K = 1"); a.legend(frameon=False, fontsize=14, ncol=2, loc="upper right")
    for a_, l in zip(ax, "abc"): panel_label(a_, l)
    fig.tight_layout(); fig.savefig(F + "Fig9.4.1_prior_fitted_incontext.png"); plt.close(fig)

SETS = [("ID", "in distribution", NAVY, "o"), ("low-nu", r"low $\nu$", BLUE, "s"),
        ("high-k", "high-$k$ input", GOLD, "^"), ("reaction", "new term (reaction)", RED, "D")]

if "uq" in which:
    d = np.load(D + "94d_uq.npz"); q = d["q"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for a, key, xl in ((ax[0], "spread", "Ensemble spread, rms sd"), (ax[1], "res", "PDE residual, rms $r$ (no labels)")):
        for n, lab, c, mk in SETS:
            a.loglog(d[f"{n}|{key}"], d[f"{n}|err"], mk, color=c, ms=5, alpha=0.55, mec="none", label=lab)
        a.set_xlabel(xl); a.set_ylabel("True rms error")
    xx = np.array([3e-4, 4e-1]); ax[0].plot(xx, xx, "k--", lw=1.6); ax[0].text(8e-2, 1.3e-1, "error = sd", fontsize=14, rotation=40)
    ax[0].set_xlim(3e-4, 4e-1); ax[0].set_ylim(5e-4, 1.5)
    xx = np.array([3e-3, 5]); ax[1].plot(xx, q[3] * xx, "k--", lw=1.6)
    ax[1].text(0.16, 6.5e-3, r"conformal bound $\hat{q}\,\|r\|$" + "\n(calibrated in distribution)", fontsize=14, ha="left", va="top")
    ax[1].set_xlim(3e-3, 5); ax[1].set_ylim(5e-4, 1.5)
    lg = ax[0].legend(frameon=False, loc="lower right", fontsize=14, markerscale=2.2, handletextpad=0.2, borderaxespad=0.1)
    for h in lg.legend_handles: h.set_alpha(1)
    pool = {k: np.concatenate([d[f"{n}|{k}"] for n, *_ in SETS]) for k in ("rel", "spread", "res")}
    f = np.linspace(0, 1, 121); rng = np.random.default_rng(0); N = len(pool["rel"])
    def curve(score):
        o = np.argsort(score); cs = np.concatenate([[0], np.cumsum(pool["rel"][o])])
        return 100 * cs[(N - np.round(f * N)).astype(int)] / N
    a = ax[2]
    a.plot(100 * f, np.mean([curve(rng.random(N)) for _ in range(20)], 0), ":", color=GRAY, lw=2.6, label="random")
    a.plot(100 * f, curve(pool["spread"]), "--", color=BLUE, lw=2.6, label="ensemble spread")
    a.plot(100 * f, curve(pool["res"]), "-", color=RED, lw=3.0, label="PDE residual")
    a.plot(100 * f, curve(pool["rel"]), "-.", color="k", lw=2.0, label="oracle (true error)")
    a.set_xlabel("Queries sent back to the solver (%)"); a.set_ylabel("Mean relative error of the hybrid (%)")
    a.set_xlim(0, 100); a.set_ylim(0, 20.5); a.legend(frameon=False, loc="upper right", fontsize=14)
    for a_, l in zip(ax, "abc"): panel_label(a_, l)
    fig.tight_layout(); fig.savefig(F + "Fig9.4.3_uq_spread_vs_residual.png"); plt.close(fig)

if "conformal" in which:
    d = np.load(D + "94d_uq.npz"); q = d["q"]; ex = np.load(D + "94d_example.npz")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    a = ax[0]; i, j = 0, 9; x = np.arange(128) * 2 * np.pi / 128
    err = np.abs(ex["true"][i, j] - ex["mu"][i, j]); sd = ex["sd"][i, j]; rms_sd = np.sqrt((ex["sd"][i] ** 2).mean())
    a.semilogy(x, err, "-", color="k", lw=2.2, label=r"$|u-\mu|$ (true error)")
    a.semilogy(x, 1.96 * sd, ":", color=BLUE, lw=2.4, label=r"$1.96\,\mathrm{sd}(x)$, uncalibrated")
    a.semilogy(x, q[0] * sd, "-", color=RED, lw=2.4, label=r"conformal, local scale")
    a.axhline(q[1] * rms_sd, color=GREEN, ls="--", lw=2.4, label="conformal, trajectory scale")
    a.set_xlabel("$x$"); a.set_ylabel("Half-width at $t = 1$"); a.set_xlim(0, 2 * np.pi); a.set_ylim(1e-5, 3e3)
    a.legend(frameon=False, fontsize=13.5, loc="upper center", ncol=1)
    a = ax[1]
    names = [n for n, *_ in SETS]
    cov = {"band, local": [(d[f"{n}|zmax"] <= q[0]).mean() for n in names],
           "band, trajectory": [(d[f"{n}|emax"] <= q[1] * d[f"{n}|spread"]).mean() for n in names],
           "norm bound, spread": [(d[f"{n}|err"] <= q[2] * d[f"{n}|spread"]).mean() for n in names],
           "norm bound, residual": [(d[f"{n}|err"] <= q[3] * d[f"{n}|res"]).mean() for n in names]}
    w = 0.2
    for k, ((lab, v), c, h) in enumerate(zip(cov.items(), (RED, GREEN, BLUE, GOLD), ("", "//", "..", "xx"))):
        a.bar(np.arange(4) + (k - 1.5) * w, 100 * np.array(v), w, color=c, hatch=h, edgecolor="white", label=lab)
    a.axhline(90, color="k", ls="--", lw=1.5); a.text(3.45, 92, "nominal 90%", fontsize=14, ha="right")
    a.set_xticks(range(4)); a.set_xticklabels(["in dist.", r"low $\nu$", "high-$k$", "reaction"])
    a.set_ylabel("Observed coverage (%)"); a.set_ylim(0, 128); a.set_yticks([0, 25, 50, 75, 100])
    a.legend(frameon=False, fontsize=13.5, ncol=2, loc="upper center", columnspacing=0.8, handlelength=1.4)
    a = ax[2]
    c_inf, c_ens, c_res, c_data, c_tr, fno_err = d["cost"]; sol = {(int(r[0]), r[1]): (r[2], r[3]) for r in d["sol"]}
    Nq = np.logspace(1, 7, 200)
    a.loglog(Nq, Nq * sol[(256, 0.005)][0] / 1e3, "-", color=NAVY, lw=2.8, label="reference solver (256 pts)")
    a.loglog(Nq, Nq * sol[(32, 0.1)][0] / 1e3, "--", color=GREEN, lw=2.6, label="solver, 32 pts (0.1% error)")
    a.loglog(Nq, c_data + c_tr + Nq * c_inf / 1e3, "-", color=RED, lw=2.8, label="FNO (1.4% error) + training")
    a.loglog(Nq, c_data + 4 * c_tr + Nq * c_ens / 1e3, ":", color=GOLD, lw=3.0, label="ensemble of 4 + training")
    ns = 1e3 * (c_data + c_tr) / (sol[(256, 0.005)][0] - c_inf)
    a.plot([ns], [ns * sol[(256, 0.005)][0] / 1e3], "o", color="k", ms=10, zorder=5)
    a.annotate(f"$N^*$ = {ns:,.0f}", (ns, ns * sol[(256, 0.005)][0] / 1e3), xytext=(25, 2.2e3), fontsize=15,
               arrowprops=dict(arrowstyle="-", color="k", lw=1.2))
    a.set_xlabel("Number of queries $N$"); a.set_ylabel("Total cost (CPU-s)"); a.set_xlim(10, 1e7); a.set_ylim(1e-2, 2e5)
    a.legend(frameon=False, fontsize=13.5, loc="lower right")
    for a_, l in zip(ax, "abc"): panel_label(a_, l)
    fig.tight_layout(); fig.savefig(F + "Fig9.4.4_conformal_breakeven.png"); plt.close(fig)

if "fewshot" in which:
    import re
    d = np.load(D + "94c_fewshot.npz"); R = {k: v for k, v in zip(d["keys"], d["vals"])}
    phys = [float(m) for m in re.findall(r"residual only, \d+ steps \(0 labels\)\s+target-test error\s+([\d.]+)",
                                          open("../ch09/labs/94e_physics_ft.out").read())]
    Ks = [4, 16, 64, 224]
    sty = [("scratch", "from scratch", GRAY, "s", "--"), ("last", "last layer (65)", GOLD, "v", ":"),
           ("lora4", "LoRA r = 4 (1,820)", RED, "o", "-"), ("full", "full fine-tuning", BLUE, "^", "-.")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    a = ax[0]
    for key, lab, c, mk, ls in sty:
        m = np.array([R[f"{K}|{key}"] for K in Ks])
        a.errorbar(Ks, m[:, 0], yerr=m[:, 1], color=c, marker=mk, ls=ls, lw=2.4, ms=9, capsize=4, label=lab)
    a.axhline(float(d["zero"]), color="k", ls=(0, (1, 1)), lw=1.8); a.text(60, float(d["zero"]) * 1.06, "zero-shot (K = 0)", fontsize=14, va="bottom")
    a.axhline(phys[1], color=GREEN, ls=(0, (5, 2)), lw=2.2); a.text(3.6, phys[1] * 0.93, "residual only, 0 labels", color=GREEN, fontsize=14, va="top", ha="left")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(Ks); a.set_xticklabels([str(k) for k in Ks]); a.minorticks_off()
    a.set_yticks([2, 5, 10, 20, 50]); a.set_yticklabels(["2", "5", "10", "20", "50"]); a.set_ylim(0.7, 90)
    a.set_xlabel("Labelled target trajectories K"); a.set_ylabel("Target-test relative error (%)")
    a.legend(frameon=False, fontsize=13.5, loc="lower left")
    a = ax[1]; i = 0; x = np.arange(128) * 2 * np.pi / 128
    pe = np.load(D + "94e_example.npy")
    a.plot(x, d["ex_true"][i, 0], color=GRAY, lw=1.4, ls=":", label="$u_0$")
    a.plot(x, d["ex_true"][i, 2], color="k", lw=3.0, label="solver, $t = 1$")
    a.plot(x, d["ex_zero"][i, 1], color=NAVY, lw=2.0, ls="--", label="zero-shot")
    a.plot(x, d["ex_lora4"][i, 1], color=RED, lw=2.0, ls="-", label="LoRA, K = 64")
    a.plot(x, pe[i, 1], color=GREEN, lw=2.0, ls="-.", label="residual only")
    a.set_xlabel("$x$"); a.set_ylabel("$u(x, t=1)$"); a.set_xlim(0, 2 * np.pi)
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.55 * (hi - lo)); a.legend(frameon=False, fontsize=13.5, ncol=2, loc="upper center")
    a = ax[2]
    for key, lab, c, mk, ls in sty[1:]:
        m = np.array([R[f"{K}|{key}"] for K in Ks])
        a.plot(Ks, m[:, 2], color=c, marker=mk, ls=ls, lw=2.4, ms=9, label=lab.split(" (")[0])
    a.axhline(0.72, color="k", ls=(0, (1, 1)), lw=1.8); a.text(4.2, 0.80, "before adaptation = LoRA switched off", fontsize=14, va="bottom")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(Ks); a.set_xticklabels([str(k) for k in Ks]); a.minorticks_off()
    a.set_yticks([1, 10, 100]); a.set_yticklabels(["1", "10", "100"]); a.set_ylim(0.4, 400)
    a.set_xlabel("Labelled target trajectories K"); a.set_ylabel("Error on the pre-training family (%)")
    a.legend(frameon=False, fontsize=13.5, loc="upper right")
    for a_, l in zip(ax, "abc"): panel_label(a_, l)
    fig.tight_layout(); fig.savefig(F + "Fig9.4.2_fewshot_adaptation.png"); plt.close(fig)
