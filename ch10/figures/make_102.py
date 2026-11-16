"""Figures for Section 10.2: Fig10.2.1 (labels), Fig10.2.2 (NVE), Fig10.2.3 (angular). Reads ../data/102_*.json."""
import sys, json, os
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, PURPLE, GRAY

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
which = sys.argv[1:] or ['labels', 'nve', 'angular']

if 'labels' in which:
    r = json.load(open(os.path.join(D, '102_stress.json')))
    Ns = [10, 40, 160]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    a.plot(Ns, [r[f'E only_N{n}']['F_mae'] for n in Ns], 'o--', color=RED, ms=10, lw=2, label='energy only')
    a.plot(Ns, [r[f'E+F_N{n}']['F_mae'] for n in Ns], 's-', color=NAVY, ms=10, lw=2, label='energy + forces')
    a.plot([40], [r['E+F+S_N40']['F_mae']], '*', color=GOLD, ms=20, mec='k', label='energy + forces + stress')
    a.axhline(997, color=GRAY, ls=':', lw=1.5); a.text(11, 1100, 'zero-force baseline', color=GRAY, fontsize=14)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xticks(Ns); a.set_xticklabels(Ns)
    a.set_xlabel('training cells $N$'); a.set_ylabel('test force MAE (meV/Å)')
    a.set_ylim(25, 1e4); a.legend(frameon=False, fontsize=13, loc='upper right')
    panel_label(a, 'a')
    tags = ['E only_N160', 'E+F_N160', 'F only_N160', 'E+F, F noise 0.2 eV/A_N160', 'E+F+S_N40']
    names = ['E\n(160)', 'E+F\n(160)', 'F\n(160)', 'E+F\nnoisy F\n(160)', 'E+F+S\n(40)']
    cols = [RED, NAVY, PURPLE, BLUE, GOLD]
    x = np.arange(len(tags))
    b = ax[1]
    b.bar(x, [r[t]['E_mae_shifted'] for t in tags], color=cols, edgecolor='k')
    for xi, t in zip(x, tags):
        b.text(xi, r[t]['E_mae_shifted'] + 0.8, f"{r[t]['E_mae_shifted']:.1f}", ha='center', fontsize=13)
    b.set_xticks(x); b.set_xticklabels(names, fontsize=13)
    b.set_ylabel('energy MAE after offset (meV/atom)')
    panel_label(b, 'b')
    c = ax[2]
    c.bar(x, [r[t]['S_mae'] for t in tags], color=cols, edgecolor='k')
    for xi, t in zip(x, tags):
        c.text(xi, r[t]['S_mae'] + 0.06, f"{r[t]['S_mae']:.2f}", ha='center', fontsize=13)
    c.set_xticks(x); c.set_xticklabels(names, fontsize=13)
    c.set_ylabel('stress MAE (GPa)')
    panel_label(c, 'c')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Fig10.2.1_labels.png')); print('labels saved')

if 'nve' in which:
    r = json.load(open(os.path.join(D, '102_nve.json')))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sty = {'energy_dt1.0': (NAVY, '-', 'conservative, 1.0 fs'), 'energy_dt0.5': (BLUE, '--', 'conservative, 0.5 fs'),
           'direct_dt1.0': (RED, '-', 'direct force, 1.0 fs'), 'direct_dt0.5': (GOLD, '--', 'direct force, 0.5 fs')}
    for k, (c, ls, lab) in sty.items():
        ax[0].plot(r[k]['t'], r[k]['E'], ls, color=c, lw=2, label=lab)
        ax[1].plot(r[k]['to'], r[k]['Eo'], ls, color=c, lw=2, marker='o', ms=6, label=lab)
    ax[0].set_xlabel('time (ps)'); ax[0].set_ylabel('$\\Delta(E_{model}+K)$ (meV/atom)')
    ax[1].set_xlabel('time (ps)'); ax[1].set_ylabel('$\\Delta(E_{oracle}+K)$ (meV/atom)')
    ax[0].legend(frameon=False, fontsize=13)
    panel_label(ax[0], 'a'); panel_label(ax[1], 'b')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Fig10.2.2_nve.png')); print('nve saved')

if 'angular' in which:
    r = json.load(open(os.path.join(D, '102_angular.json')))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    rcs = sorted({k.split('_')[1] for k in r})
    x = np.arange(len(rcs)); w = 0.36
    for j, (q, lab) in enumerate([('F_mae', 'force MAE (meV/Å)'), ('E_mae', 'energy MAE (meV/atom)')]):
        a = ax[j]
        vi = [r[f'inv_{rc}_N400'][q] for rc in rcs]; ve = [r[f'eq_{rc}_N400'][q] for rc in rcs]
        a.bar(x - w / 2, vi, w, color=RED, edgecolor='k', hatch='//', label='invariant ($l=0$)')
        a.bar(x + w / 2, ve, w, color=NAVY, edgecolor='k', label='equivariant ($l\\leq1$)')
        for xi, v in zip(x - w / 2, vi): a.text(xi, v * 1.02, f'{v:.0f}' if q == 'F_mae' else f'{v:.1f}', ha='center', fontsize=14)
        for xi, v in zip(x + w / 2, ve): a.text(xi, v * 1.02, f'{v:.0f}' if q == 'F_mae' else f'{v:.1f}', ha='center', fontsize=14)
        a.set_xticks(x); a.set_xticklabels([f"$r_c$ = {rc[2:]} Å" for rc in rcs])
        a.set_ylabel(lab)
        if j == 0: a.legend(frameon=False, fontsize=14)
        panel_label(a, 'ab'[j])
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Fig10.2.3_angular.png')); print('angular saved')
