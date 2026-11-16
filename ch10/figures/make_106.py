"""Figures for Section 10.6: Fig10.6.1 (relaxation, ROI) and Fig10.6.2 (MACE dissection)."""
import os, sys, json
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data')
R = json.load(open(os.path.join(DATA, '106_relax.json')))
D = json.load(open(os.path.join(DATA, '106_dissect.json')))

# ---------------------------------------------------------------- Fig10.6.1
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(wspace=0.32))
for tag, c, ls, mk, lab in [('sym-constrained', NAVY, '-', 'o', 'cubic, symmetry fixed'),
                             ('free, rattled', RED, '--', 's', 'rattled, no constraint')]:
    h = R['A'][tag]['hist']
    ax[0].semilogy(range(len(h)), h, ls=ls, marker=mk, color=c, ms=6, lw=2, label=lab)
ax[0].axhline(1e-3, color=GRAY, ls=':', lw=1.5); ax[0].text(8, 1.25e-3, 'fmax = 0.001 eV/Å', color=GRAY, fontsize=14)
ax[0].set_xlabel('BFGS step'); ax[0].set_ylabel('max generalized force (eV/Å)')
ax[0].legend(frameon=False, loc='upper right'); panel_label(ax[0], 'a')

cases = list(R['C'].keys()); x = np.arange(len(cases)); w = 0.36
raw = [R['C'][c]['ref_from_raw'] for c in cases]; pre = [R['C'][c]['ref_from_mace'] for c in cases]
mc = [R['C'][c]['mace_steps'] for c in cases]
b1 = ax[1].bar(x - w / 2, raw, w, color=NAVY, label='reference from raw start')
b2 = ax[1].bar(x + w / 2, pre, w, color=GOLD, edgecolor='k', hatch='//', label='reference after MACE pre-relax')
for xi, v in zip(x - w / 2, raw): ax[1].text(xi, v + 0.8, str(v), ha='center', fontsize=15)
for xi, v, m in zip(x + w / 2, pre, mc): ax[1].text(xi, v + 0.8, f'{v}\n(+{m} MACE)', ha='center', fontsize=13)
ax[1].set_xticks(x); ax[1].set_xticklabels(['SrTiO$_3$ (5 atoms,\ncell + positions)', 'Si vacancy\n(63 atoms)'])
ax[1].set_ylabel('reference force calls'); ax[1].set_ylim(0, max(raw) * 1.35)
ax[1].legend(frameon=False, loc='upper left', fontsize=14); panel_label(ax[1], 'b')
fig.savefig(os.path.join(HERE, 'Fig10.6.1_relaxation_roi.png')); plt.close(fig)

# ---------------------------------------------------------------- Fig10.6.2
fig, ax = plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw=dict(wspace=0.42, width_ratios=[1.05, 1.1, 1.1]))
A = D['A']; tot = A['total']
other = tot - A['skip'] - A['symc'] - A['radial'] - A['lin']
parts = [('element-dependent\nskip TP', A['skip'], NAVY), ('symmetric\ncontractions', A['symc'], BLUE),
         ('equivariant\nlinears', A['lin'], GREEN), ('radial MLPs', A['radial'], GOLD), ('other', other, GRAY)]
y = np.arange(len(parts))[::-1]
ax[0].barh(y, [p[1] / 1e6 for p in parts], color=[p[2] for p in parts])
for yi, p in zip(y, parts): ax[0].text(p[1] / 1e6 + 0.04, yi, f'{100*p[1]/tot:.1f}%', va='center', fontsize=15)
ax[0].set_yticks(y); ax[0].set_yticklabels([p[0] for p in parts], fontsize=14)
ax[0].set_xlabel('parameters (millions)'); ax[0].set_xlim(0, 3.6); panel_label(ax[0], 'a', x=-0.5)

z = np.load(os.path.join(DATA, '106_receptive.npz')); r, dF, de = z['r'], z['dF'], z['de']
m = r > 0.1
ax[1].semilogy(r[m], np.maximum(de[m], 1e-17), 'o', color=NAVY, ms=6, label=r'$|\Delta\varepsilon_i|$ (eV)')
ax[1].semilogy(r[m], np.maximum(dF[m], 1e-17), 's', color=RED, ms=5, mfc='none', mew=1.4, label=r'$|\Delta\mathbf{F}_i|$ (eV/Å)')
for xv in (6, 12): ax[1].axvline(xv, color=GRAY, ls='--', lw=1.4)
ax[1].text(6.2, 3e-16, '$r_c$', fontsize=16, color=GRAY); ax[1].text(12.2, 3e-16, '$2r_c$', fontsize=16, color=GRAY)
ax[1].set_ylim(1e-17, 1); ax[1].set_xlabel('distance from the moved atom (Å)'); ax[1].set_ylabel('change after a 0.01 Å move')
ax[1].legend(frameon=False, loc='upper right', fontsize=14); panel_label(ax[1], 'b')

n = np.arange(1, 7); w = 0.27
for k, (key, c, h, lab) in enumerate([('layer1', NAVY, '', 'layer-1 energy'), ('layer2', RED, '//', 'layer-2 energy'),
                                       ('LJ', GOLD, '..', 'Lennard-Jones')]):
    v = np.maximum(np.abs(np.array(D['D'][key])), 1e-17)
    ax[2].bar(n + (k - 1) * w, v, w, color=c, hatch=h, edgecolor='k', lw=0.6, label=lab, log=True)
ax[2].axhspan(1e-17, 1e-13, color=GRAY, alpha=0.15, label='round-off level')
ax[2].set_ylim(1e-17, 1e9); ax[2].set_xticks(n); ax[2].set_xlabel('body order n'); ax[2].set_ylabel(r'$|E_n|$ (eV)')
ax[2].legend(frameon=False, loc='upper center', ncol=2, fontsize=12, columnspacing=0.8, handlelength=1.5); panel_label(ax[2], 'c')
fig.savefig(os.path.join(HERE, 'Fig10.6.2_mace_dissection.png')); plt.close(fig)
print('saved')
