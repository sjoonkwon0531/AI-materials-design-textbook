"""Fig10.1.1: cost of one energy+force call versus N, and the break-even of the MLIP route (data/101_cost.json)."""
import sys, json, os
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import plt, panel_label, NAVY, RED, BLUE, GOLD, GREEN, GRAY

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, '..', 'data', '101_cost.json')))
Nd = np.array(sorted(int(k) for k in r['dft'])); td = np.array([r['dft'][str(n)]['cpu'] for n in Nd])
Nm = np.array(sorted(int(k) for k in r['mlip'])); tm = np.array([r['mlip'][str(n)] for n in Nm])
Ne = np.array(sorted(int(k) for k in r['emt'])); te = np.array([r['emt'][str(n)] for n in Ne])
p = r['summary']['slope_dft']; N0, t0 = Nd[-1], td[-1]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
NN = np.logspace(np.log10(N0), 3, 50)
a.plot(NN, t0 * (NN / N0) ** 3, '--', color=RED, lw=2.2, label='DFT model, $N^3$')
a.plot(NN, t0 * (NN / N0) ** p, ':', color=RED, lw=2.2, label=f'DFT model, $N^{{{p:.2f}}}$ (measured slope)')
a.plot(Nd, td, 'o', color=RED, ms=10, label='DFT measured (PySCF, PBE, SZV, $\\Gamma$)')
a.plot(Nm, tm, 's-', color=NAVY, ms=9, lw=2, label='MACE-MP-0 small measured')
a.plot(Ne, te, '^-', color=GREEN, ms=9, lw=2, label='EMT measured (Cu)')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('number of atoms $N$'); a.set_ylabel('CPU time per energy+force call (s)')
a.set_ylim(3e-3, 1e13)
a.legend(loc='upper left', frameon=False, fontsize=13)
panel_label(a, 'a')

b = ax[1]
N = 64
c_ml = np.interp(N, Nm, tm)
build = 400 * t0 + 100.0
S = np.logspace(-1, 6, 200)
b.plot(S, S * t0 * (N / N0) ** 3, '--', color=RED, lw=2.2, label='AIMD, $N^3$ model')
b.plot(S, S * t0 * (N / N0) ** p, ':', color=RED, lw=2.2, label=f'AIMD, $N^{{{p:.2f}}}$ model')
b.plot(S, build + S * c_ml, '-', color=NAVY, lw=2.5, label='MLIP: 400 labels + training + MD')
b.axhline(build, color=GRAY, lw=1, ls='-.')
b.text(2e4, build * 0.35, 'build cost', color=GRAY, fontsize=14)
b.set_xscale('log'); b.set_yscale('log')
b.set_xlabel('MD steps for $N=64$ Si'); b.set_ylabel('total CPU time (s)')
b.legend(loc='upper left', frameon=False, fontsize=13)
panel_label(b, 'b')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'Fig10.1.1_cost_scaling.png'))
print('saved')
