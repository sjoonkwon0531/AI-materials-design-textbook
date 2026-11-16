"""Figures for Section 10.7: Fig10.7.1 (phonon dispersions at a = 3.944 A) and Fig10.7.2 (tilt and lattice constant)."""
import os, sys, json
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data')
MODELS = ['mace', 'chgnet', 'sevennet', 'mattersim']
LAB = {'mace': 'MACE-MP-0 small', 'chgnet': 'CHGNet 0.3.0', 'sevennet': 'SevenNet-0', 'mattersim': 'MatterSim-v1.0.0-1M'}
COL = {'mace': NAVY, 'chgnet': RED, 'sevennet': BLUE, 'mattersim': GREEN}
LS = {'mace': '-', 'chgnet': '--', 'sevennet': '-.', 'mattersim': ':'}
MK = {'mace': 'o', 'chgnet': 's', 'sevennet': '^', 'mattersim': 'D'}
D = {m: json.load(open(os.path.join(DATA, f'107_{m}.json'))) for m in MODELS}
PBE_G = [-133, 146, 226, 508]; PBE_R = [-86, 128, 413, 433, 798]   # Evarestov et al. 2011, PW PBE, a = 3.944 A

# ---------------------------------------------------------------- Fig10.7.1
labels = ['Γ', 'X', 'M', 'Γ', 'R', 'X|M', 'R']
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), sharey=True, gridspec_kw=dict(hspace=0.28, wspace=0.08))
for ax, m, pl in zip(axs.flat, MODELS, 'abcd'):
    segs = D[m]['ph']['pbe']['band']; x0 = 0; ticks = [0]
    for seg in segs:
        seg = np.array(seg); xs = x0 + np.linspace(0, 1, len(seg))
        ax.plot(xs, seg, color=COL[m], lw=1.6)
        x0 += 1; ticks.append(x0)
    ax.axhline(0, color='k', lw=0.8)
    ax.plot([0] * 4, PBE_G, 'o', mfc='none', mec='k', ms=9, mew=1.6, label='PBE (Evarestov et al.)')
    ax.plot([4] * 5, PBE_R, 'o', mfc='none', mec='k', ms=9, mew=1.6)
    ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_xlim(0, ticks[-1])
    for t in ticks: ax.axvline(t, color=GRAY, lw=0.6)
    ax.text(0.98, 0.985, LAB[m], transform=ax.transAxes, va='top', ha='right', fontsize=17, color=COL[m], bbox=dict(fc='white', ec='none', pad=1))
    if pl in 'ac': ax.set_ylabel('frequency (cm$^{-1}$)')
    ax.set_ylim(-170, 960); panel_label(ax, pl, x=-0.12 if pl in 'ac' else -0.04)
fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=1, fontsize=15, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(HERE, 'Fig10.7.1_sto_phonons.png')); plt.close(fig)

# ---------------------------------------------------------------- Fig10.7.2
def soft(fr, k):
    f = np.array(fr[k])
    if k == 'G':                                   # drop the three acoustic modes (|f| ~ 0)
        f = np.delete(f, np.argsort(np.abs(f))[:3])
    return f.min()


fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw=dict(wspace=0.3))
for m in MODELS:
    t = D[m]['tilt']
    ax[0].plot(t['phis'], t['prof'], ls=LS[m], marker=MK[m], color=COL[m], ms=5, lw=2, label=LAB[m])
    ax[0].plot(t['phi_rel'], t['dE_rel'], marker='*', color=COL[m], ms=17, mec='k', ls='none')
ax[0].axvline(4.9, color='k', ls='--', lw=1.3); ax[0].text(5.05, 9.2, 'PBE 4.9°', fontsize=14)
ax[0].axvline(2.1, color=GRAY, ls=':', lw=1.6); ax[0].text(0.15, 9.2, 'exp. 2.1°', fontsize=14, color=GRAY)
ax[0].axhline(0, color='k', lw=0.7)
ax[0].set_xlim(0, 8); ax[0].set_ylim(-5, 11)
ax[0].set_xlabel('octahedral tilt angle (degree)'); ax[0].set_ylabel('E − E(cubic) (meV per f.u.)')
ax[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.9), fontsize=13); panel_label(ax[0], 'a')

for m in MODELS:
    a = [D[m]['ph'][k]['a'] for k in ('exp', 'pbe', 'own')]
    g = [soft(D[m]['ph'][k]['freqs'], 'G') for k in ('exp', 'pbe', 'own')]
    rr = [soft(D[m]['ph'][k]['freqs'], 'R') for k in ('exp', 'pbe', 'own')]
    o = np.argsort(a); a, g, rr = np.array(a)[o], np.array(g)[o], np.array(rr)[o]
    ax[1].plot(a, g, ls='-', marker=MK[m], color=COL[m], ms=8, lw=2)
    ax[1].plot(a, rr, ls='--', marker=MK[m], color=COL[m], ms=8, lw=2, mfc='white')
ax[1].plot([3.944], [-133], 'o', mfc='none', mec='k', ms=12, mew=2); ax[1].plot([3.944], [-86], 'o', mfc='none', mec='k', ms=12, mew=2)
ax[1].text(3.947, -140, 'PBE Γ', fontsize=13); ax[1].text(3.947, -80, 'PBE R', fontsize=13)
ax[1].axhline(0, color='k', lw=0.7); ax[1].axvline(3.905, color=GRAY, ls=':', lw=1.5)
ax[1].text(3.906, 155, 'exp. a', color=GRAY, fontsize=14)
ax[1].plot([], [], 'k-', lw=2, label='lowest Γ mode (solid)'); ax[1].plot([], [], 'k--', lw=2, label='lowest R mode (dashed)')
ax[1].legend(frameon=False, loc='upper right', fontsize=13)
ax[1].set_xlabel('lattice constant a (Å)'); ax[1].set_ylabel('soft-mode frequency (cm$^{-1}$)')
ax[1].set_ylim(-160, 175); panel_label(ax[1], 'b')
fig.savefig(os.path.join(HERE, 'Fig10.7.2_tilt_lattice.png')); plt.close(fig)
print('saved')
