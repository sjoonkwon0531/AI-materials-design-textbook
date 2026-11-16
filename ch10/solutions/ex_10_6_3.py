"""Exercise 10.6.3 (c): Si monovacancy formation energy with MACE-MP-0 small in a 512-atom cell (4x4x4),
same recipe as labs/106_relax.py Part B (positions only, fmax 0.005 eV/A, four neighbours rattled by 0.02 A,
seed 1). float32 is checked against float64 on the 64-atom cell first.
Run: /tmp/claude-0/v10/bin/python ex_10_6_3.py"""
import time, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(1)
from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from mace.calculators import MACECalculator
MODEL = '../ch10/data/mace_mp0_small.model'

def bulk_si(calc):
    si = bulk('Si', 'diamond', a=5.43, cubic=True); si.calc = calc; si.set_constraint(FixSymmetry(si))
    BFGS(FrechetCellFilter(si), logfile=None).run(fmax=1e-4, steps=200)
    a = si.cell.cellpar()[0]; e = si.get_potential_energy() / len(si); si.set_constraint()
    return si, a, e

def vacancy(si0, calc, rep, e_si):
    sc = si0.repeat((rep, rep, rep)); N = len(sc)
    vac = sc.copy(); del vac[0]; vac.calc = calc; Eu = vac.get_potential_energy()
    dist0 = np.linalg.norm(((sc.get_scaled_positions()[1:] + 0.5) % 1 - 0.5) @ sc.cell, axis=1)
    nbr = np.argsort(dist0)[:4]
    pert = vac.copy(); rng = np.random.default_rng(1)
    pert.positions[nbr] += rng.normal(0, 0.02, (4, 3)); pert.calc = calc
    t0 = time.process_time(); opt = BFGS(pert, logfile=None); opt.run(fmax=0.005, steps=400)
    cpu = time.process_time() - t0
    Er = pert.get_potential_energy()
    dd = pert.get_all_distances(mic=True)[np.ix_(nbr, nbr)]; pairs = sorted(dd[np.triu_indices(4, 1)])
    return N, Eu - (N - 1) * e_si, Er - (N - 1) * e_si, opt.nsteps, cpu, pairs

t00 = time.process_time()
for dtype, reps in (('float64', (2,)), ('float32', (2, 4))):
    calc = MACECalculator(model_paths=MODEL, device='cpu', default_dtype=dtype)
    si0, a, e = bulk_si(calc)
    print(f"{dtype}: bulk a = {a:.4f} A, E/atom = {e:.5f} eV")
    for rep in reps:
        N, Efu, Efr, n, cpu, pairs = vacancy(si0, calc, rep, e)
        print(f"  N = {N:3d}: Ef(unrelaxed) {Efu:.3f} eV, Ef(relaxed) {Efr:.3f} eV, steps {n}, CPU {cpu:.0f} s;"
              f" nbr-nbr {' '.join(f'{x:.3f}' for x in pairs)} A", flush=True)
print(f"CPU total {time.process_time() - t00:.0f} s")
