"""104_val.py -- Lab for Section 10.4.3: a validation report card beyond the held-out error.

Part A (leakage): one student trained on 24 random frames of ONE 1500 K oracle trajectory,
  evaluated on (i) held-out frames of the same trajectory that lie within 2 steps (4 fs) of a
  training frame, (ii) a time block of the same trajectory not touched by training, and
  (iii) an independent 1500 K trajectory.
Part B (report card): the three final students of Lab 104_al (qbc, rand, aimd) and the oracle on
  force RMSE (1500 K, 2500 K, 1800 K, strained cells, vacancy cell), EOS (a0, B), Gamma optical phonon,
  relaxed vacancy formation energy, and the 2500 K radial distribution function (student MD 1 ps vs oracle MD).
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')
from ase import Atoms, units
from ase.build import bulk
from ase.eos import EquationOfState
from ase.optimize import FIRE

t0 = time.process_time()
pool = C.load_frames(os.path.join(C.DATA, '104_pool.npz'))
test = C.load_frames(os.path.join(C.DATA, '104_test.npz'))
sets = dict(T1500=test, T2500=C.load_frames(os.path.join(C.DATA, '104_test_2500K.npz')),
            T1800=C.load_frames(os.path.join(C.DATA, '104_transfer_1800K.npz')),
            strain=C.load_frames(os.path.join(C.DATA, '104_transfer_strain.npz')),
            vacancy=C.load_frames(os.path.join(C.DATA, '104_transfer_vacancy.npz')))

# ---------------- Part A: leakage through time-correlated frames
traj = [f for f in pool if f['T'] == 1500 and f['step'] > 20]           # 180 consecutive frames, 2 fs apart
rng = np.random.default_rng(0)
first = np.arange(0, 120)                                                 # training frames drawn from steps 21..140
tr_idx = np.sort(rng.choice(first, 24, replace=False))
near = [k for k in first if k not in tr_idx and np.abs(tr_idx - k).min() <= 2]
block = list(range(140, 180))                                             # last 80 fs, never used for training
e0 = np.mean([f['E'] / 64 for f in traj])
m = C.new_student(0, e0); C.train(m, [traj[k] for k in tr_idx], epochs=120, seed=0)
eA = dict(near=C.errors(m, [traj[k] for k in near]), block=C.errors(m, [traj[k] for k in block]), indep=C.errors(m, test))
print("Part A, one student trained on 24 frames of a single 1500 K oracle trajectory:")
print(f"  held-out frames within 4 fs of a training frame ({len(near)}): F-RMSE {eA['near'][1]:6.1f} meV/A, E {eA['near'][0]:5.2f} meV/atom")
print(f"  later time block of the same trajectory (40):        F-RMSE {eA['block'][1]:6.1f} meV/A, E {eA['block'][0]:5.2f} meV/atom")
print(f"  independent 1500 K trajectory (21):                  F-RMSE {eA['indep'][1]:6.1f} meV/A, E {eA['indep'][0]:5.2f} meV/atom")

# ---------------- Part B: report card
orc = C.oracle('float32')
calcs = {'oracle': orc}
for arm in ('qbc', 'rand', 'aimd'):
    calcs[arm] = C.StudentCalc(C.load_student(os.path.join(C.DATA, f'104_al_{arm}.pt')))
ref_frames = [f for f in C.load_frames(os.path.join(C.DATA, '104_pool_hot.npz')) if f['T'] == 2500 and f['step'] > 50]


def rdf(frames, rmax=5.0, nb=100):
    from ase.neighborlist import primitive_neighbor_list
    h = np.zeros(nb); edges = np.linspace(0, rmax, nb + 1)
    for f in frames:
        d = primitive_neighbor_list('d', [True] * 3, f['cell'], f['pos'], rmax)
        h += np.histogram(d, edges)[0]
    V = abs(np.linalg.det(frames[0]['cell'])); n = len(frames[0]['Z'])
    r = 0.5 * (edges[1:] + edges[:-1]); shell = 4 * np.pi * r ** 2 * (edges[1] - edges[0])
    return r, h / (len(frames) * n * (n / V) * shell)


def gamma_phonon(calc):
    """Gamma optical frequency: rigid shift of one fcc sublattice along x, central difference of forces."""
    a = bulk('Si', 'diamond', a=C.A0, cubic=True); a.calc = calc
    frac = a.get_scaled_positions()
    sub = [i for i in range(8) if np.allclose((frac[i] * 4) % 2, 0)]      # atoms on the (0,0,0) fcc sublattice
    u = 0.01; Fx = []
    for s in (+1, -1):
        b = a.copy(); b.calc = calc; p = b.positions.copy(); p[sub, 0] += s * u; b.positions = p
        Fx.append(b.get_forces()[sub, 0].mean())
    k = -(Fx[0] - Fx[1]) / (2 * u)                                          # eV/A^2 per atom of the sublattice
    mu = 28.0855 / 2 * units._amu                                            # reduced mass of the two sublattices
    return np.sqrt(k * units._e * 1e20 / mu) / (2 * np.pi) / 1e12           # THz


tt = {}
for name in ('oracle', 'qbc'):
    a = C.si64(); a.rattle(0.05, seed=1); a.calc = calcs[name]; a.get_forces()
    t = time.process_time()
    for k in range(10):
        a.positions += 0.001; a.get_forces()
    tt[name] = (time.process_time() - t) / 10
print(f"CPU time per 64-atom force call: oracle (MACE-MP-0 small, float32) {tt['oracle']:.3f} s, "
      f"student SchNet {tt['qbc'] * 1000:.1f} ms, ratio {tt['oracle'] / tt['qbc']:.0f}x")

card = {}
for name, calc in calcs.items():
    row = {}
    if name != 'oracle':
        mdl = calc.model
        for k, s in sets.items():
            row[k] = C.errors(mdl, s)[1]
    V, E = [], []
    for s in np.linspace(5.38, 5.56, 7):
        a = bulk('Si', 'diamond', a=s, cubic=True).repeat((2, 2, 2)); a.calc = calc
        V.append(a.get_volume()); E.append(a.get_potential_energy())
    v0, _, B = EquationOfState(V, E).fit()
    row['a0'] = (v0 / 8) ** (1 / 3); row['B'] = B / units.GPa
    row['nu_gamma'] = gamma_phonon(calc)
    perf = C.si64(); perf.calc = calc; Ep = perf.get_potential_energy()
    vac = C.si64(); del vac[0]; vac.calc = calc
    FIRE(vac, logfile=None).run(fmax=0.02, steps=300)
    row['Evac'] = vac.get_potential_energy() - 63 / 64 * Ep
    if name == 'oracle':
        r, g = rdf(ref_frames)
    else:
        base = C.si64()
        fr, fail = C.langevin(calc.model, base.positions, base.cell.array, base.numbers, 2500, 500, seed=77, every=2)
        row['md_fail'] = fail
        r, g = rdf([f for f in fr if f['step'] > 100]) if len(fr) > 60 else (None, None)
    row['rdf'] = None if g is None else g.tolist()
    card[name] = row
    print(f"{name:6s}: " + (" | ".join(f"F[{k}] {row[k]:5.0f}" for k in sets) + " meV/A | " if name != 'oracle' else "") +
          f"a0 {row['a0']:.4f} A, B {row['B']:5.1f} GPa, nu_Gamma {row['nu_gamma']:5.2f} THz, E_vac {row['Evac']:.2f} eV" +
          (f", MD fail step {row.get('md_fail')}" if name != 'oracle' else ""))
g_ref = np.array(card['oracle']['rdf'])
for name in ('qbc', 'rand', 'aimd'):
    if card[name]['rdf'] is not None:
        g = np.array(card[name]['rdf']); i1 = np.argmax(g_ref)
        print(f"  RDF {name}: first-peak height {g[np.argmax(g)]:.2f} at {r[np.argmax(g)]:.3f} A (oracle {g_ref[i1]:.2f} at {r[i1]:.3f} A); "
              f"integrated |g - g_ref| dr = {np.abs(g - g_ref).sum() * (r[1] - r[0]):.3f} A")
card['r'] = r.tolist(); card['partA'] = eA
json.dump(card, open(os.path.join(C.DATA, '104_val.json'), 'w'), default=float)
print(f"CPU {time.process_time() - t0:.0f} s")
