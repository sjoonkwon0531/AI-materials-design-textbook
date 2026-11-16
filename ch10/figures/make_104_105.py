"""Figures for Sections 10.4 and 10.5 (owner C).  Run with any python that has matplotlib."""
import sys, json, os
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import matplotlib.pyplot as plt

D = '../ch10/data'
OUT = '../ch10/figures'
which = sys.argv[1:] or ['data', 'al', 'val', 'ft']


def fig_data():
    d = np.load(f'{D}/104_data_fig.npz'); s = json.load(open(f'{D}/104_data_summary.json'))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw=dict(width_ratios=[1, 1.1, 1.2]))
    lag = 2 * np.arange(len(d['acf300']))
    for T, c, ls, mk in [(300, BLUE, '-', 'o'), (900, GOLD, '--', 's'), (1500, RED, '-.', '^')]:
        ax[0].plot(lag, d[f'acf{T}'], color=c, ls=ls, marker=mk, ms=5, markevery=4, lw=2, label=f'{T} K')
    ax[0].axhline(0, color=GRAY, lw=0.8); ax[0].axhline(0.1, color=GRAY, lw=0.8, ls=':')
    ax[0].set_xlabel('lag (fs)'); ax[0].set_ylabel('force autocorrelation'); ax[0].legend(frameon=False)
    P, T = d['P'], d['T']
    for t, c, mk in [(300, BLUE, 'o'), (900, GOLD, 's'), (1500, RED, '^')]:
        m = T == t; ax[1].scatter(P[m, 0], P[m, 1], s=10, color=c, marker=mk, alpha=0.35, label=f'MD {t} K')
    ax[1].scatter(d['Pr'][:, 0], d['Pr'][:, 1], s=40, marker='x', color=GREEN, label='rattled')
    ax[1].scatter(P[d['fps'], 0], P[d['fps'], 1], s=110, facecolor='none', edgecolor=NAVY, lw=2, label='FPS picks')
    ax[1].scatter(P[d['blk'], 0], P[d['blk'], 1], s=70, marker='D', facecolor='none', edgecolor=PURPLE, lw=1.6, label='block picks')
    ax[1].scatter(d['Pt'][:, 0], d['Pt'][:, 1], s=40, marker='*', color='k', label='test 1500 K')
    ax[1].set_xlabel('SOAP PC1'); ax[1].set_ylabel('SOAP PC2'); ax[1].legend(frameon=False, fontsize=12, loc='best', ncol=1)
    names = ['block', 'random', 'fps', 'rattle']; labs = ['block', 'random', 'FPS', 'rattle']
    x = np.arange(len(keys)); w = 0.26
    for k, (col, c, h, lab) in enumerate([(0, RED, '', '1500 K test'), (1, PURPLE, '//', '1800 K'), (2, BLUE, '..', 'strained')]):
        ax[2].bar(x + (k - 1) * w, [s[n]['mean'][col] for n in names], w, yerr=[s[n]['std'][col] for n in names],
                  color=c, hatch=h, edgecolor='white', capsize=3, label=lab, alpha=0.9)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labs); ax[2].set_ylabel('force RMSE (meV/Å)'); ax[2].set_ylim(0, 330)
    ax[2].legend(frameon=False, ncol=3, loc='upper center', fontsize=13, handlelength=1.4, columnspacing=0.8)
    for a, l in zip(ax, 'abc'):
        panel_label(a, l)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig10.4.1_sampling.png'); plt.close(fig)


def fig_al():
    AS = [json.load(open(f'{D}/104_al.json')), json.load(open(f'{D}/104_al_s1.json'))]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    sty = dict(qbc=(NAVY, '-', 'o', 'QbC (committee)'), rand=(RED, '--', 's', 'random from student MD'),
               aimd=(GREEN, '-.', '^', 'random from oracle MD'))
    for arm, (c, ls, mk, lab) in sty.items():
        n = [r['labels'] for r in AS[0][arm]]
        F = np.array([[r['F_rmse'] for r in A[arm]] for A in AS])
        st = np.array([[sum([A[arm][k]['stable_steps'] for A in AS], []) for k in range(len(n))]])[0] * 2e-3
        calls = [r['oracle_calls'] for r in AS[0][arm]]
        ax[0].plot(n, F.mean(0), color=c, ls=ls, marker=mk, ms=8, lw=2, label=lab)
        for f in F:
            ax[0].plot(n, f, color=c, ls='none', marker=mk, ms=4, alpha=0.35)
        ax[1].plot(n, st.mean(1), color=c, ls=ls, marker=mk, ms=8, lw=2, label=lab)
        ax[1].fill_between(n, st.min(1), st.max(1), color=c, alpha=0.12)
        ax[2].plot(calls, st.mean(1), color=c, ls=ls, marker=mk, ms=8, lw=2, label=lab)
    ax[0].set_xlabel('oracle labels'); ax[0].set_ylabel('force RMSE, 2500 K test (meV/Å)')
    ax[1].set_xlabel('oracle labels'); ax[1].set_ylabel('stable MD time at 2500 K (ps)'); ax[1].set_ylim(0, 1.08)
    ax[2].set_xlabel('total oracle calls (labels + oracle MD)'); ax[2].set_ylabel('stable MD time at 2500 K (ps)')
    ax[2].set_xscale('log'); ax[2].set_ylim(0, 1.08)
    ax[0].legend(frameon=False, fontsize=13)
    for a, l in zip(ax, 'abc'):
        panel_label(a, l)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig10.4.2_active_learning.png'); plt.close(fig)


def fig_val():
    V = json.load(open(f'{D}/104_val.json')); r = np.array(V['r'])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))
    keys = ['T1500', 'T1800', 'T2500', 'strain', 'vacancy']; labs = ['1500 K', '1800 K', '2500 K', 'strained', 'vacancy']
    x = np.arange(len(keys)); w = 0.26
    for k, (arm, c, h) in enumerate([('qbc', NAVY, ''), ('rand', RED, '//'), ('aimd', GREEN, '..')]):
        ax[0].bar(x + (k - 1) * w, [V[arm][kk] for kk in keys], w, color=c, hatch=h, edgecolor='white',
                  label={'qbc': 'QbC', 'rand': 'random (student MD)', 'aimd': 'random (oracle MD)'}[arm])
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylabel('force RMSE (meV/Å)'); ax[0].legend(frameon=False)
    ax[1].plot(r, V['oracle']['rdf'], color='k', lw=2.5, label='oracle')
    for arm, c, ls in [('qbc', NAVY, '--'), ('rand', RED, ':'), ('aimd', GREEN, '-.')]:
        if V[arm]['rdf'] is not None:
            ax[1].plot(r, V[arm]['rdf'], color=c, ls=ls, lw=2, label={'qbc': 'QbC', 'rand': 'random (student MD)', 'aimd': 'random (oracle MD)'}[arm])
    ax[1].set_xlim(1.8, 5.0); ax[1].set_xlabel('r (Å)'); ax[1].set_ylabel('g(r), 2500 K'); ax[1].legend(frameon=False)
    for a, l in zip(ax, 'ab'):
        panel_label(a, l)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig10.4.3_validation.png'); plt.close(fig)


def fig_ft():
    R = json.load(open(f'{D}/105_ft.json')); res = R['results']
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    sty = {'zero-shot': (GRAY, ':', 'D'), 'readout': (BLUE, '--', 's'), 'full': (NAVY, '-', 'o'), 'scratch': (RED, '-.', '^')}
    for m, (c, ls, mk) in sty.items():
        rr = [x for x in res if x['method'] == m]; N = [x['N'] for x in rr]
        ax[0].plot(N, [x['F'] for x in rr], color=c, ls=ls, marker=mk, ms=8, lw=2, label=m)
        ax[1].plot(N, [x['B'] for x in rr], color=c, ls=ls, marker=mk, ms=8, lw=2, label=m)
    f2 = [x for x in res if x['method'] == 'full-200'][0]
    for a, v in zip(ax[:2], [f2['F'], f2['B']]):
        a.plot([48], [v], marker='*', ms=20, color=GOLD, mec=NAVY, ls='none', label='full, 200 steps')
    ax[0].axhline(R['raw']['F'], color=GRAY, lw=1); ax[0].axhline(R['schnet']['F'], color=GOLD, lw=2, ls='--')
    ax[0].text(4.2, R['schnet']['F'] * 1.06, 'SchNet, scratch, 300 epochs', color=GOLD, ha='left', fontsize=13)
    ax[1].axhline(R['ref']['B_GPa'], color='k', lw=2); ax[1].text(4.2, R['ref']['B_GPa'] + 1.5, 'GFN2-xTB', fontsize=13)
    ax[1].axhline(R['raw']['B'], color=GRAY, lw=1); ax[1].text(4.2, R['raw']['B'] + 1.5, 'MACE-MP-0 small', fontsize=13, color=GRAY)
    for a in ax[:2]:
        a.set_xscale('log'); a.set_xticks([4, 16, 48]); a.set_xticklabels(['4', '16', '48']); a.set_xlabel('xTB labels (8-atom cells)')
    mats = ['C', 'SiC', 'Ge', 'Al', 'NaCl']; xx = np.arange(5); w = 0.26
    for k, (m, c, h, lab) in enumerate([('readout', BLUE, '//', 'readout'), ('full', NAVY, '', 'full'), ('full-200', GOLD, '..', 'full, 200 steps')]):
        x = [r for r in res if r['method'] == m and r['N'] == 48][0]
        ax[2].bar(xx + (k - 1) * w, [x['drift'][q] for q in mats], w, color=c, hatch=h, edgecolor='white', label=lab)
    ax[2].set_xticks(xx); ax[2].set_xticklabels(mats); ax[2].set_xlabel('material (not in fine-tuning data)')
    ax[0].set_yscale('log'); ax[0].set_ylabel('force RMSE vs xTB (meV/Å)'); ax[1].set_ylabel('bulk modulus (GPa)')
    ax[2].set_ylabel('force drift vs MACE-MP-0 (meV/Å)'); ax[2].set_yscale('log')
    ax[2].legend(frameon=False, fontsize=13, loc='upper right'); ax[2].set_ylim(1, 2000); ax[0].legend(frameon=False, fontsize=12, loc='center', bbox_to_anchor=(0.5, 0.76), ncol=2)
    for a, l in zip(ax, 'abc'):
        panel_label(a, l)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig10.5.1_finetune.png'); plt.close(fig)


for w in which:
    {'data': fig_data, 'al': fig_al, 'val': fig_val, 'ft': fig_ft}[w]()
