"""Figures for Section 10.3 (reads data/103_<model>.json written by labs/103_bench.py and 103_inspect.py)."""
import os, sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data')
MODELS = ['mace', 'chgnet', 'sevennet', 'mattersim']
LAB = {'mace': 'MACE-MP-0 small', 'chgnet': 'CHGNet 0.3.0', 'sevennet': 'SevenNet-0', 'mattersim': 'MatterSim 1M'}
COL = {'mace': NAVY, 'chgnet': RED, 'sevennet': BLUE, 'mattersim': GREEN}
MK = {'mace': 'o', 'chgnet': 's', 'sevennet': '^', 'mattersim': 'D'}
LS = {'mace': '-', 'chgnet': '--', 'sevennet': '-.', 'mattersim': ':'}
R = {m: json.load(open(os.path.join(DATA, f'103_{m}.json'))) for m in MODELS}
ELEMS = ['Al', 'Cu', 'Ag', 'Au', 'Si', 'Ge', 'Mo', 'W']

# ---------- Fig 10.3.1: parameters vs receptive field
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for m in MODELS:
    a = R[m]['arch']; p = R[m]['info']['n_params']
    ax[0].scatter(p, a['receptive_field_A'], s=160, marker=MK[m], color=COL[m], label=LAB[m], zorder=3)
    k = MODELS.index(m); v = a['atoms_in_receptive_field_Si']
    ax[1].bar(k, v, color=COL[m], hatch=['', '//', '..', 'xx'][k], edgecolor='white')
    ax[1].text(k, v + 60, f'{v:,}', ha='center', va='bottom', fontsize=16)
ax[0].set_xscale('log'); ax[0].set_xlabel('Number of parameters'); ax[0].set_ylabel('Receptive field  $T r_c$ (Å)')
ax[0].set_xlim(2e5, 8e6); ax[0].set_ylim(0, 30); ax[0].legend(frameon=False, loc='lower left', fontsize=14)
ax[0].set_xticks([3e5, 1e6, 3e6]); ax[0].set_xticklabels(['3×10$^5$', '10$^6$', '3×10$^6$'])
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([LAB[m].replace(' ', '\n', 1) for m in MODELS], fontsize=14)
ax[1].set_ylabel('Si atoms inside receptive field'); ax[1].set_ylim(0, 3700)
panel_label(ax[0], 'a'); panel_label(ax[1], 'b')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Fig10.3.1_arch_map.png')); plt.close(fig)

# ---------- Fig 10.3.2: EOS benchmark
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
x = np.arange(len(ELEMS)); w = 0.18
for k, m in enumerate(MODELS):
    dv = [100 * (R[m]['eos'][e]['V0'] / R[m]['eos'][e]['V0_pbe'] - 1) for e in ELEMS]
    rb = [R[m]['eos'][e]['B0'] / R[m]['eos'][e]['B0_pbe'] for e in ELEMS]
    ax[0].bar(x + (k - 1.5) * w, dv, w, color=COL[m], label=LAB[m])
    ax[1].plot(x, rb, marker=MK[m], ls=LS[m], color=COL[m], ms=9, lw=1.6, label=LAB[m])
ex = [100 * (R['mace']['eos'][e]['V0_exp'] / R['mace']['eos'][e]['V0_pbe'] - 1) for e in ELEMS]
eb = [R['mace']['eos'][e]['B0_exp'] / R['mace']['eos'][e]['B0_pbe'] for e in ELEMS]
ax[0].scatter(x, ex, marker='_', s=500, color=GRAY, lw=3, zorder=4, label='Experiment')
ax[1].scatter(x, eb, marker='*', s=160, color=GRAY, zorder=4, label='Experiment')
for a_ in ax:
    a_.set_xticks(x); a_.set_xticklabels(ELEMS)
ax[0].axhline(0, color='k', lw=0.8); ax[1].axhline(1, color='k', lw=0.8)
ax[0].set_ylabel('$V_0$ deviation from PBE (%)'); ax[1].set_ylabel('$B_0$ / $B_0^{\\mathrm{PBE}}$')
ax[1].set_ylim(0.6, 1.4)
panel_label(ax[0], 'a'); panel_label(ax[1], 'b')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=15, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Fig10.3.2_eos_benchmark.png')); plt.close(fig)

# ---------- Fig 10.3.3: NVE, dimer, timing
fig, ax = plt.subplots(1, 3, figsize=(17, 4.9))
for m in MODELS:
    n = R[m]['nve']; t = np.arange(len(n['E'])) * n['dt_fs'] / 1000
    ax[0].plot(t, n['E'], ls=LS[m], color=COL[m], lw=1.8, label=LAB[m])
    d = R[m]['ood']['Si2']
    ax[1].plot(d['r'], d['E'], ls=LS[m], marker=MK[m], ms=5, color=COL[m], lw=1.6, label=LAB[m])
    tm = R[m]['timing']; N = [int(k) for k in tm]
    ax[2].plot(N, [tm[k]['us_per_atom'] / 1e3 for k in tm], ls=LS[m], marker=MK[m], ms=9, color=COL[m], lw=1.6)
ax[0].set_xlabel('Time (ps)'); ax[0].set_ylabel('$E_{\\mathrm{tot}}(t)-E_{\\mathrm{tot}}(0)$ (meV/atom)')
ax[0].legend(frameon=False, fontsize=13, ncol=2, loc='upper left'); ax[0].set_ylim(-0.02, 0.62)
ax[1].set_xlabel('Si–Si distance (Å)'); ax[1].set_ylabel('Dimer energy (eV)'); ax[1].set_ylim(-16, 60); ax[1].set_xlim(0.55, 4.05)
ax[1].axhline(0, color='k', lw=0.6)
ax[2].set_xscale('log'); ax[2].set_xlabel('Number of atoms'); ax[2].set_ylabel('CPU time per atom (ms)')
ax[2].set_xticks([8, 64, 216]); ax[2].set_xticklabels(['8', '64', '216'])
for k, l in enumerate('abc'): panel_label(ax[k], l)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'Fig10.3.3_dynamics_ood_cost.png')); plt.close(fig)
print('figures written')
