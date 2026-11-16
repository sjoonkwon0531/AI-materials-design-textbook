"""Ch.1 v1.2 figures: Fig 1.3.1 (new), Fig 1.5.1 and Fig 1.6.1 (restyled).
Figure rules: Arial Narrow (Liberation Sans Narrow is the metric-compatible
fallback used in this environment), large fonts, panel labels a/b only,
no panel titles, Midnight Executive palette, English labels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

NAVY = "#1E2761"; ICE = "#CADCFC"; GOLD = "#F4B942"
RED = "#C0392B"; BLUE = "#2471A3"; GRAY = "#6B6B6B"

FONT = ["Arial Narrow", "Liberation Sans Narrow"]
plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans Narrow",
    "mathtext.it": "Liberation Sans Narrow:italic",
    "mathtext.bf": "Liberation Sans Narrow:bold",
    "font.size": 17,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def panel_label(ax, letter, x=-0.14, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------- Fig 1.3.1
def fig_bias_variance():
    c = np.linspace(0.05, 1.0, 400)          # model complexity (arbitrary)
    bias2 = 0.9 * np.exp(-4.2 * c)
    var = 0.03 * np.exp(3.2 * c) - 0.02
    noise = np.full_like(c, 0.12)
    total = bias2 + var + noise
    i = np.argmin(total)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(c, total, color=NAVY, lw=3.0)
    ax.plot(c, bias2, color=BLUE, lw=2.4, ls="--")
    ax.plot(c, var, color=RED, lw=2.4, ls="-.")
    ax.plot(c, noise, color=GRAY, lw=1.8, ls=":")
    ax.plot(c[i], total[i], "o", ms=12, mfc=GOLD, mec=NAVY, mew=1.6, zorder=5)

    # direct labels (secondary encoding besides color and dash style)
    ax.text(c[-60] - 0.02, total[-60] + 0.02, "total error", color=NAVY, ha="right", va="bottom")
    ax.text(c[60] - 0.06, bias2[60] - 0.07, r"bias$^2$", color=BLUE, va="top")
    ax.text(c[340] + 0.02, var[340], "variance", color=RED, ha="left", va="center")
    ax.text(0.98, 0.135, r"noise $\sigma^2$", color=GRAY, ha="right", va="bottom")
    ax.annotate("optimal complexity", xy=(c[i], total[i]),
                xytext=(c[i], total[i] + 0.3), ha="center",
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.3))
    ax.text(0.07, -0.08, "underfitting", transform=ax.transAxes, color=GRAY, fontsize=16)
    ax.text(0.97, -0.08, "overfitting", transform=ax.transAxes, color=GRAY,
            ha="right", fontsize=16)

    ax.set_xlabel("Model complexity", labelpad=22)
    ax.set_ylabel("Expected prediction error")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0, 1.0); ax.set_xlim(0.03, 1.0)
    fig.tight_layout()
    fig.savefig("Fig1.3.1_bias_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Fig1.3.1 saved; optimum at complexity", round(c[i], 3))


# ---------------------------------------------------------------- Fig 1.5.1
def fig_truncated_svd():
    yr = np.array([1930, 1940, 1949, 1960, 1970, 1980, 1990, 2000, 2010], float)
    pop = np.array([2044, 2355, 2017, 2499, 3144, 3741, 4339, 4599, 4799], float)
    A = np.column_stack([yr ** 2, yr, np.ones_like(yr)])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    def coeffs(t):
        sinv = np.array([1 / si if k < t else 0.0 for k, si in enumerate(s)])
        return Vt.T @ np.diag(sinv) @ U.T @ pop

    xx = np.linspace(1925, 2015, 300)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    styles = {3: (NAVY, "-", 3.0), 2: (RED, "--", 2.6), 1: (BLUE, ":", 2.8)}
    ends = {}
    for t in (3, 2, 1):
        X = coeffs(t)
        yy = X[0] * xx ** 2 + X[1] * xx + X[2]
        col, ls, lw = styles[t]
        ax.plot(xx, yy, ls, color=col, lw=lw)
        ends[t] = yy[-1]
        print(f"t={t}: RMS={np.sqrt(np.mean((A @ X - pop) ** 2)):.2f}")
    ax.plot(yr, pop, "o", ms=10, mfc=GOLD, mec=NAVY, mew=1.5, zorder=5)

    ax.text(2016, ends[3] + 60, "full SVD, t = 3", color=NAVY, va="center")
    ax.text(2016, ends[2] - 120, "truncated, t = 2", color=RED, va="center")
    ax.text(2016, ends[1], "truncated, t = 1", color=BLUE, va="center")
    ax.text(1931, 4450, "data", color=NAVY)
    ax.plot([1928], [4520], "o", ms=10, mfc=GOLD, mec=NAVY, mew=1.5)

    ax.set_xlim(1925, 2045)
    ax.set_xticks([1930, 1950, 1970, 1990, 2010])
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Population ($10^4$)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("Fig1.5.1_truncated_svd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Fig1.5.1 saved")


# ---------------------------------------------------------------- Fig 1.6.1
def fig_newton_fractal():
    roots = np.array([1.0, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)])
    N, span, max_iter, tol = 1000, 2.0, 50, 1e-6
    x = np.linspace(-span, span, N)
    X, Y = np.meshgrid(x, x)
    Zk = X + 1j * Y
    basin = np.full(Zk.shape, -1, dtype=int)
    itc = np.zeros(Zk.shape, dtype=int)
    active = np.ones(Zk.shape, dtype=bool)
    for k in range(max_iter):
        df = 3.0 * Zk ** 2
        safe = np.abs(df) > 1e-12
        step = np.zeros_like(Zk)
        step[safe] = (Zk[safe] ** 3 - 1.0) / df[safe]
        Zk = Zk - step
        for ri, r in enumerate(roots):
            hit = active & (np.abs(Zk - r) < tol)
            basin[hit] = ri; itc[hit] = k + 1; active[hit] = False
        if not active.any():
            break
    itc[active] = max_iter
    print("converged", np.mean(basin >= 0), "mean it", round(np.mean(itc[basin >= 0]), 3),
          "fractions", [round(np.mean(basin == i), 4) for i in range(3)], "max", itc.max())

    rgb = np.ones(Zk.shape + (3,))
    shade_it = np.clip(itc / 18.0, 0, 1)
    for i, col in enumerate([NAVY, GOLD, RED]):
        m = basin == i
        base = np.array(to_rgb(col))
        sh = (0.45 + 0.55 * shade_it[m])[:, None]
        rgb[m] = base[None, :] * sh + (1 - sh)
    rgb[basin < 0] = to_rgb(ICE)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    ax = axes[0]
    ax.imshow(rgb, extent=[-span, span, -span, span], origin="lower")
    for i, r in enumerate(roots):
        ax.plot(r.real, r.imag, "o", mfc="white", mec="black", mew=1.6, ms=11)
        ax.annotate(f"$z_{i + 1}$", (r.real, r.imag), textcoords="offset points",
                    xytext=(9, 7), fontsize=20, fontweight="bold")
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
    panel_label(ax, "a", x=-0.16)

    ax = axes[1]
    im = ax.imshow(itc, extent=[-span, span, -span, span], origin="lower",
                   cmap="magma", vmin=0, vmax=18)
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Newton iterations to converge")
    panel_label(ax, "b", x=-0.16)
    for a in axes:
        a.spines["top"].set_visible(True); a.spines["right"].set_visible(True)
    fig.tight_layout()
    fig.savefig("Fig1.6.1_newton_fractal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Fig1.6.1 saved")


if __name__ == "__main__":
    fig_bias_variance()
    fig_truncated_svd()
    fig_newton_fractal()
