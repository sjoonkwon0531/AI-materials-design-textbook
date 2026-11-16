import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from bookstyle import *
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch, Polygon
L = '../../ch05/labs/'; F = '../../ch05/figures/'

# ---------------- Fig 5.6.1 schematic
fig, axs = plt.subplots(1, 4, figsize=(18, 5.0))
rng = np.random.default_rng(3)
# (a) QM/MM
ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10)
pts = rng.uniform(0.4, 9.6, (110, 2))
for p in pts:
    if np.hypot(*(p-5)) > 2.3: ax.add_patch(Circle(p, 0.22, color=GRAY, alpha=0.6))
ax.add_patch(Circle((5, 5), 2.2, color=GOLD, alpha=0.35, lw=0))
for p in [(4.3, 5.3), (5.6, 5.5), (5.0, 4.2), (4.4, 4.4), (5.8, 4.5)]:
    ax.add_patch(Circle(p, 0.3, color=RED))
ax.plot([5.6, 6.9], [5.5, 6.3], '-', color=NAVY, lw=2); ax.add_patch(Circle((6.45, 6.03), 0.16, color=BLUE))
ax.text(5, 7.5, 'QM', ha='center', fontsize=20, color=RED, weight='bold', bbox=dict(fc='white', ec='none', pad=1)); ax.text(0.6, 0.0, 'MM', fontsize=20, color=GRAY, weight='bold', bbox=dict(fc='white', ec='none', pad=1))
ax.text(6.6, 6.7, 'link atom', fontsize=14, color=BLUE, bbox=dict(fc='white', ec='none', pad=1))
# (b) AdResS
ax = axs[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.add_patch(Rectangle((3.5, 0.6), 3, 8.4, color=GOLD, alpha=0.25, lw=0))
ax.add_patch(Rectangle((4.4, 0.6), 1.2, 8.4, color=RED, alpha=0.18, lw=0))
for p in rng.uniform([0.3, 1.0], [9.7, 8.8], (40, 2)):
    x = p[0]
    if 4.4 <= x <= 5.6:      # atomistic: 3-site molecules
        th = rng.uniform(0, np.pi)
        for k in (-1, 0, 1): ax.add_patch(Circle((x+0.25*k*np.cos(th), p[1]+0.25*k*np.sin(th)), 0.14, color=RED))
    elif 3.5 <= x <= 6.5:    # hybrid
        ax.add_patch(Circle(p, 0.33, color=PURPLE, alpha=0.35)); ax.add_patch(Circle(p, 0.12, color=RED))
    else:
        ax.add_patch(Circle(p, 0.33, color=BLUE, alpha=0.7))
xs = np.linspace(0, 10, 200); w = np.clip(1-(np.abs(xs-5)-0.6)/0.9, 0, 1)
ax.plot(xs, 9.2+0.6*w, '-', color=NAVY, lw=2.2); ax.text(0.1, 9.35, 'w(x)', fontsize=15, color=NAVY)
ax.text(1.3, 0.0, 'CG', fontsize=18, color=BLUE, weight='bold'); ax.text(4.25, 0.0, 'AT', fontsize=18, color=RED, weight='bold')
ax.text(7.8, 0.0, 'CG', fontsize=18, color=BLUE, weight='bold')
# (c) HMM
ax = axs[2]; ax.set_xlim(0, 10); ax.set_ylim(0, 10)
xg = np.linspace(0.8, 9.2, 9)
ax.plot([0.8, 9.2], [7.5, 7.5], '-', color=NAVY, lw=2.5); ax.plot(xg, np.full(9, 7.5), 'o', color=NAVY, ms=9)
ax.text(0.5, 8.4, 'macro solver  (unknown D, flux)', fontsize=15, color=NAVY)
for xm in (xg[2], xg[6]):
    ax.add_patch(FancyBboxPatch((xm-1.1, 1.4), 2.2, 2.6, boxstyle='round,pad=0.05', fc='white', ec=GREEN, lw=2))
    yy = np.linspace(1.6, 3.8, 60); ax.plot(xm-0.9+1.8*(0.5+0.4*np.sin(2*np.pi*yy*1.6)), yy, color=GOLD, lw=1.5)
    t = np.cumsum(rng.choice([-1, 1], 80))*0.04; ax.plot(xm+t[:60]*0.6, np.linspace(1.6, 3.8, 60), color=GREEN, lw=1)
    ax.add_patch(FancyArrowPatch((xm-0.3, 7.2), (xm-0.3, 4.2), arrowstyle='-|>', mutation_scale=18, color=GRAY, lw=1.6))
    ax.add_patch(FancyArrowPatch((xm+0.3, 4.2), (xm+0.3, 7.2), arrowstyle='-|>', mutation_scale=18, color=GREEN, lw=1.6))
ax.text(xg[2]-1.4, 5.5, 'constrain', fontsize=13, color=GRAY, rotation=90, va='center')
ax.text(xg[2]+0.55, 5.5, 'estimate', fontsize=13, color=GREEN, rotation=90, va='center')
ax.text(5, 0.3, 'micro solvers (local, short)', ha='center', fontsize=15, color=GREEN)
# (d) quasicontinuum
ax = axs[3]; ax.set_xlim(0, 10); ax.set_ylim(0, 10)
X, Y = np.meshgrid(np.arange(0.5, 10, 0.5), np.arange(0.5, 10, 0.5))
core = np.hypot(X-5, Y-5) < 1.8
ax.plot(X[core], Y[core], 'o', color=RED, ms=5)
ax.plot(X[~core], Y[~core], '.', color=GRAY, ms=2.5, alpha=0.5)
rep = [(1, 1), (5, 1), (9, 1), (1, 5), (9, 5), (1, 9), (5, 9), (9, 9), (3, 3), (7, 3), (3, 7), (7, 7), (5, 3), (3, 5), (7, 5), (5, 7)]
tri = [((1,1),(5,1),(3,3)), ((5,1),(9,1),(7,3)), ((1,1),(1,5),(3,3)), ((9,1),(9,5),(7,3)), ((1,5),(1,9),(3,7)),
       ((9,5),(9,9),(7,7)), ((1,9),(5,9),(3,7)), ((5,9),(9,9),(7,7)), ((5,1),(3,3),(5,3)), ((5,1),(7,3),(5,3)),
       ((1,5),(3,3),(3,5)), ((1,5),(3,7),(3,5)), ((9,5),(7,3),(7,5)), ((9,5),(7,7),(7,5)), ((5,9),(3,7),(5,7)), ((5,9),(7,7),(5,7))]
for t in tri: ax.add_patch(Polygon(t, closed=True, fill=False, ec=NAVY, lw=1.4))
xr, yr = zip(*rep); ax.plot(xr, yr, 's', color=NAVY, ms=8)
ax.text(5, 9.6, 'representative atoms + FE interpolation', ha='center', fontsize=14, color=NAVY)
ax.text(5, 5, '', ha='center')
for k, ax in enumerate(axs):
    ax.set_aspect('equal'); ax.axis('off'); panel_label(ax, 'abcd'[k], x=-0.02, y=1.0)
fig.subplots_adjust(wspace=0.08)
fig.savefig(F+'Fig5.6.1_bridging_schematic.png'); plt.close(fig)

# ---------------- Fig 5.6.2 HMM demo
m = np.load(L+'56_micro.npz'); h = np.load(L+'56_hmm.npz')
fig, axs = plt.subplots(1, 3, figsize=(17, 4.9), gridspec_kw=dict(wspace=0.33))
ax = axs[0]; ax.semilogx(m['t']/(1/64)**2*1.0, m['D_t'], 'o-', color=NAVY, ms=4, lw=1.5, label='micro MSD/2t')
ax.axhline(float(m['arith']), color=RED, ls='-', lw=2, label='arithmetic (Voigt-like)')
ax.axhline(float(m['harm']), color=BLUE, ls='--', lw=2, label='harmonic (Reuss, exact)')
ax.set_xlabel(r'micro time  $t\,\bar D/\varepsilon^2$'); ax.set_ylabel(r'$D_{\rm eff}$ estimate'); ax.set_ylim(0.3, 1.05)
ax.legend(frameon=False, fontsize=13, loc='center right', bbox_to_anchor=(1.0, 0.68)); panel_label(ax, 'a', x=-0.2)
ax = axs[1]; x = h['xf_macro']
ax.plot(x, h['D_ari'], '-', color=RED, lw=2, label='arithmetic')
ax.plot(x, h['D_hom'], '--', color=BLUE, lw=2, label='homogenized')
ax.plot(x, h['D_hmm'], 'o', color=GOLD, mec='k', ms=7, label='HMM (micro KMC)')
ax.set_xlabel('x'); ax.set_ylabel(r'interface $D(x)$'); ax.legend(frameon=False, fontsize=13, loc='upper left'); panel_label(ax, 'b', x=-0.2)
ax = axs[2]
for key, ls, lab in (('01', '-', 't = 0.01'), ('1', '-', 't = 0.1')):
    ax.plot(h['xc_f'], h['fine'+key], ls, color=NAVY, lw=1.6, label='fine, '+lab)
    ax.plot(h['xc_m'], h['hmm'+key], 'o', color=GOLD, mec='k', ms=6, label='HMM' if key == '01' else None)
    ax.plot(h['xc_m'], h['ari'+key], 's', color=RED, mfc='none', ms=6, label='arithmetic' if key == '01' else None)
ax.annotate('t = 0.01', xy=(0.09, 0.2), xytext=(0.66, 0.33), fontsize=15, arrowprops=dict(arrowstyle='->', color=GRAY))
ax.annotate('t = 0.1', xy=(0.28, 0.2), xytext=(0.56, 0.5), fontsize=15, arrowprops=dict(arrowstyle='->', color=GRAY))
ax.set_xlabel('x'); ax.set_ylabel('u(x, t)'); ax.set_xlim(0, 1)
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0], handles[1], handles[2]], ['fine-scale truth', 'HMM', 'arithmetic'], frameon=False, fontsize=13, loc='upper right')
panel_label(ax, 'c', x=-0.2)
fig.savefig(F+'Fig5.6.2_hmm.png'); plt.close(fig)
print('done')
