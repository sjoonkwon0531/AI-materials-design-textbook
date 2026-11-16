"""Figures for section 6.2 (data from labs/62_cm.py and labs/62_desc.py)."""
import sys, importlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
sys.path.insert(0, "../ch06/labs")
from bookstyle import *  # noqa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

L = "../ch06/labs/"
F = "../ch06/figures/"

# ---------------- Fig6.2.1 Coulomb matrix failure modes
d = np.load(L + "62_cm_scan.npz")
s, srt, eig = d["s"], d["srt"], d["eig"]
i0 = 0
ds = np.linalg.norm(srt - srt[i0], axis=1) / np.linalg.norm(srt[i0])
de = np.linalg.norm(eig - eig[i0], axis=1) / np.linalg.norm(eig[i0])
t = np.load(L + "62_cm_twins.npz")
R0, R1 = t["R0"], t["R1"]


def dl(R):
    ch = np.sort(np.linalg.norm(R[1:] - R[0], axis=1))
    hh = np.sort([np.linalg.norm(R[i] - R[j]) for i in range(1, 5) for j in range(i + 1, 5)])
    return ch, hh


def cm(Z, R):
    D = np.linalg.norm(R[:, None] - R[None], axis=-1); np.fill_diagonal(D, 1)
    M = np.outer(Z, Z) / D; np.fill_diagonal(M, 0.5 * Z ** 2.4); return M


Zm = np.array([6., 1, 1, 1, 1])
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
ax = axs[0]
ax.plot(s, ds, "-", color=RED, lw=2.4, label="sorted-row CM")
ax.plot(s, de, "--", color=NAVY, lw=2.4, label="eigenvalue CM")
ax.axvline(0, color=GRAY, lw=1, ls=":")
ax.set_xlabel(r"O$-$H stretch relative to crossing, $s-s^{*}$ ($\AA$)")
ax.set_ylabel("relative distance to start")
ax.legend(frameon=False, loc="upper left")
panel_label(ax, "a")
ax = axs[1]
ch0, hh0 = dl(R0); ch1, hh1 = dl(R1)
x = np.arange(10)
ax.plot(x, np.r_[ch0, hh0], "o-", color=NAVY, ms=9, lw=1.8, label="geometry 1")
ax.plot(x, np.r_[ch1, hh1], "s--", color=RED, ms=8, lw=1.8, label="geometry 2")
ax.axvline(3.5, color=GRAY, lw=1, ls=":")
ax.text(1.5, 2.05, "C$-$H", ha="center", fontsize=16); ax.text(6.5, 2.05, "H$-$H", ha="center", fontsize=16)
ax.set_xticks(x); ax.set_xticklabels([])
ax.set_xlabel("sorted interatomic distances"); ax.set_ylabel(r"distance ($\AA$)")
ax.set_ylim(0.6, 2.2)
ax.legend(frameon=False, loc="lower right")
panel_label(ax, "b")
ax = axs[2]
e0 = np.sort(np.linalg.eigvalsh(cm(Zm, R0)))[::-1]; e1 = np.sort(np.linalg.eigvalsh(cm(Zm, R1)))[::-1]
k = np.arange(1, 6)
ax.plot(k, e0, "o", color=NAVY, ms=14, mfc="none", mew=2.2, label="geometry 1")
ax.plot(k, e1, "x", color=RED, ms=11, mew=2.4, label="geometry 2")
ax.set_xticks(k); ax.set_xlabel("eigenvalue index"); ax.set_ylabel(r"$\lambda_k$ of CM")
ax.legend(frameon=False)
panel_label(ax, "c")
fig.tight_layout()
fig.savefig(F + "Fig6.2.1_cm_failures.png"); plt.close(fig)

# ---------------- Fig6.2.2 SOAP construction, 2D cartoon
rng = np.random.default_rng(4)
nb = np.array([[1.0, 0.2], [-0.6, 0.9], [-0.7, -0.75], [0.3, -1.05], [1.6, 1.3], [-1.75, 0.2]])
X, Y = np.meshgrid(np.linspace(-2.3, 2.3, 300), np.linspace(-2.3, 2.3, 300))
rc = 2.0; sig = 0.25
rho = np.zeros_like(X)
for p in nb:
    r = np.hypot(*p)
    fc = 0.5 * (np.cos(np.pi * r / rc) + 1) if r < rc else 0.0
    rho += fc * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * sig ** 2))
fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
ax = axs[0]
ax.plot(0, 0, "o", color=RED, ms=18)
ax.plot(nb[:, 0], nb[:, 1], "o", color=NAVY, ms=14)
ax.add_patch(plt.Circle((0, 0), rc, fill=False, ls="--", color=GRAY, lw=1.6))
ax.text(0.18, 0.12, "$i$", fontsize=20, color=RED)
ax.text(1.25, -1.75, r"$r_{\mathrm{cut}}$", fontsize=18, color=GRAY)
ax.set_aspect("equal"); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.axis("off")
panel_label(ax, "a", x=0.0)
ax = axs[1]
cmap = LinearSegmentedColormap.from_list("w2n", ["white", ICE, BLUE, NAVY])
ax.contourf(X, Y, rho, levels=20, cmap=cmap)
ax.add_patch(plt.Circle((0, 0), rc, fill=False, ls="--", color=GRAY, lw=1.6))
ax.plot(0, 0, "o", color=RED, ms=10)
ax.set_aspect("equal"); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.axis("off")
ax.text(-2.2, 2.0, r"$\rho_i(\mathbf{r})$", fontsize=20)
panel_label(ax, "b", x=0.0)
ax = axs[2]
r = np.linspace(0, rc, 300)
for n, (c, ls) in enumerate(zip([NAVY, RED, BLUE, GREEN], ["-", "--", "-.", ":"])):
    g = r ** n * np.exp(-1.6 * r ** 2) * (1 - r / rc) ** 2
    ax.plot(r, g / np.abs(g).max(), ls, color=c, lw=2.4, label=f"$g_{n+1}(r)$")
ax.set_xlabel("$r$ (arb. units)"); ax.set_ylabel("radial basis (scaled)")
ax.legend(frameon=False, loc="upper right")
panel_label(ax, "c")
fig.tight_layout()
fig.savefig(F + "Fig6.2.2_soap_cartoon.png"); plt.close(fig)

# ---------------- Fig6.2.3 smoothness
d = np.load(L + "62_desc_smooth.npz")
amps = d["amps"]
names = ["SOAP", "ACSF", "MBTR k=2", "MBTR k=3", "sine, eigenvalues", "sine, sorted rows"]
keys = sorted([k for k in d.files if k.startswith("rel_")], key=lambda k: int(k.split("_")[-1]))
sty = [("o-", NAVY), ("s--", BLUE), ("^-.", GREEN), ("v:", PURPLE), ("D-", GOLD), ("x-", RED)]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
ax = axs[0]
for k, nm, (m, c) in zip(keys, names, sty):
    ax.loglog(amps, d[k], m, color=c, lw=1.8, ms=7, label=nm)
ax.loglog(amps, amps * 1.5, color=GRAY, lw=1.2, ls=(0, (1, 3)))
ax.text(2.2e-3, 1.1e-3, "slope 1", color=GRAY, fontsize=15)
ax.set_xlabel(r"RMS displacement per atom ($\AA$)")
ax.set_ylabel(r"$\|D(x+a u)-D(x)\| / \|D(x)\|$")
ax.legend(frameon=False, fontsize=13, loc="lower right")
panel_label(ax, "a")
ax = axs[1]
sc = d["scan"]
for i, (nm, (m, c)) in enumerate(zip(["SOAP", "sine, eigenvalues", "sine, sorted rows"],
                                       [("-", NAVY), ("--", GOLD), ("-", RED)])):
    ax.semilogy(sc, d[f"speed_{i}"], m, color=c, lw=1.6, label=nm)
ax.set_xlabel(r"position along path ($\AA$)")
ax.set_ylabel(r"$\|dD/ds\| / \|D\|$  ($\AA^{-1}$)")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(F + "Fig6.2.3_smoothness.png"); plt.close(fig)

# ---------------- Fig6.2.4 dimension and cost
importlib.import_module("62_dscribe_compat")
from dscribe.descriptors import SOAP, ACSF, MBTR
rows = np.load(L + "62_desc_cost.npz")["rows"]
fig, axs = plt.subplots(1, 3, figsize=(17, 5.0))
ax = axs[0]
for lmax, (m, c) in zip([2, 6, 10], [("o-", NAVY), ("s--", RED), ("^-.", BLUE)]):
    sel = rows[rows[:, 1] == lmax]
    ax.plot(sel[:, 0], sel[:, 2], m, color=c, ms=8, lw=2, label=rf"$l_{{\max}}={lmax}$")
ax.set_xlabel(r"$n_{\max}$"); ax.set_ylabel("SOAP length (1 species)")
ax.legend(frameon=False)
panel_label(ax, "a")
ax = axs[1]
ax.plot(rows[:, 2], rows[:, 3], "o", color=NAVY, ms=8, alpha=0.85)
cc = np.polyfit(rows[:, 2], rows[:, 3], 1)
xx = np.linspace(0, rows[:, 2].max() * 1.05, 10)
ax.plot(xx, np.polyval(cc, xx), "--", color=GRAY, lw=1.5)
ax.set_xlabel("SOAP length"); ax.set_ylabel("time per atom (ms)")
panel_label(ax, "b")
ax = axs[2]
g2 = [[eta, 0.0] for eta in [0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0, 4.0]]
g4 = [[0.005, z, lam] for z in [1, 2, 4, 16] for lam in [-1, 1]]
sp = ["Cu", "Ni", "Al", "Fe", "Co", "Cr"]
S = np.arange(1, 7)
dims = {"SOAP": [], "ACSF": [], "MBTR k=2": [], "MBTR k=3": []}
for n in S:
    dims["SOAP"].append(SOAP(species=sp[:n], r_cut=5.0, n_max=8, l_max=6, sigma=0.5).get_number_of_features())
    dims["ACSF"].append(ACSF(species=sp[:n], r_cut=5.0, g2_params=g2, g4_params=g4).get_number_of_features())
    dims["MBTR k=2"].append(MBTR(species=sp[:n], geometry={"function": "inverse_distance"},
                                 grid={"min": 0.1, "max": 0.6, "n": 200, "sigma": 0.01}).get_number_of_features())
    dims["MBTR k=3"].append(MBTR(species=sp[:n], geometry={"function": "cosine"},
                                 grid={"min": -1, "max": 1, "n": 200, "sigma": 0.02}).get_number_of_features())
for (nm, v), (m, c) in zip(dims.items(), [("o-", NAVY), ("s--", BLUE), ("^-.", GREEN), ("v:", PURPLE)]):
    ax.semilogy(S, v, m, color=c, ms=8, lw=2, label=nm)
ax.set_xlabel("number of species $S$"); ax.set_ylabel("feature length")
ax.legend(frameon=False, fontsize=14)
panel_label(ax, "c")
print("species dims:", {k: v for k, v in dims.items()})
print(f"linear fit time(ms/atom) = {cc[0]*1e3:.3f}e-3 * dim + {cc[1]:.3f}")
fig.tight_layout()
fig.savefig(F + "Fig6.2.4_dim_cost.png"); plt.close(fig)

# ---------------- Fig6.2.5 kernel heat map
d = np.load(L + "62_desc_kernel.npz")
K, nm = d["K"], [str(x) for x in d["names"]]
fig, ax = plt.subplots(figsize=(10.5, 8.8))
cmap = LinearSegmentedColormap.from_list("k", ["white", ICE, BLUE, NAVY])
im = ax.imshow(K, cmap=cmap, vmin=0.45, vmax=1.0)
for i in range(len(nm)):
    for j in range(len(nm)):
        ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=11,
                color="white" if K[i, j] > 0.8 else "black")
ax.set_xticks(range(len(nm))); ax.set_xticklabels(nm, rotation=60, ha="right", fontsize=14)
ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=14)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"SOAP kernel $k=(\hat p\cdot\hat p')^2$")
fig.savefig(F + "Fig6.2.5_soap_kernel.png"); plt.close(fig)
print("done")
