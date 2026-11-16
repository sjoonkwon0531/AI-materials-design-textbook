"""Ex 10.4.6: Lab 10.8 report card for a student saved by ex_10_4_qbc.py.
Usage: python ex_10_4_6_eval.py MODEL.pt   (v10 interpreter). Same items and settings as labs/104_val.py Part B."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, '../ch10/labs')
from importlib import import_module
C = import_module('104_common')
torch.set_num_threads(1)
from ase import units
from ase.build import bulk
from ase.eos import EquationOfState
from ase.optimize import FIRE
t0 = time.process_time()
sets = dict(T1500=C.load_frames(os.path.join(C.DATA, '104_test.npz')),
            T2500=C.load_frames(os.path.join(C.DATA, '104_test_2500K.npz')),
            T1800=C.load_frames(os.path.join(C.DATA, '104_transfer_1800K.npz')),
            strain=C.load_frames(os.path.join(C.DATA, '104_transfer_strain.npz')),
            vacancy=C.load_frames(os.path.join(C.DATA, '104_transfer_vacancy.npz')))
mdl = C.load_student(sys.argv[1]); calc = C.StudentCalc(mdl)


def gamma_phonon(calc):
    a = bulk('Si', 'diamond', a=C.A0, cubic=True); a.calc = calc
    frac = a.get_scaled_positions()
    sub = [i for i in range(8) if np.allclose((frac[i] * 4) % 2, 0)]
    u = 0.01; Fx = []
    for s in (+1, -1):
        b = a.copy(); b.calc = calc; p = b.positions.copy(); p[sub, 0] += s * u; b.positions = p
        Fx.append(b.get_forces()[sub, 0].mean())
    k = -(Fx[0] - Fx[1]) / (2 * u); mu = 28.0855 / 2 * units._amu
    return np.sqrt(k * units._e * 1e20 / mu) / (2 * np.pi) / 1e12


row = {k: C.errors(mdl, s)[1] for k, s in sets.items()}
V, E = [], []
for s in np.linspace(5.38, 5.56, 7):
    a = bulk('Si', 'diamond', a=s, cubic=True).repeat((2, 2, 2)); a.calc = calc
    V.append(a.get_volume()); E.append(a.get_potential_energy())
v0, _, B = EquationOfState(V, E).fit()
perf = C.si64(); perf.calc = calc; Ep = perf.get_potential_energy()
vac = C.si64(); del vac[0]; vac.calc = calc; FIRE(vac, logfile=None).run(fmax=0.02, steps=300)
base = C.si64()
_, fail = C.langevin(mdl, base.positions, base.cell.array, base.numbers, 2500, 500, seed=77, every=2)
print(f"{os.path.basename(sys.argv[1])}: " + " | ".join(f"F[{k}] {v:5.0f}" for k, v in row.items()) +
      f" meV/A | a0 {(v0/8)**(1/3):.4f} A, B {B/units.GPa:5.1f} GPa, nu_Gamma {gamma_phonon(calc):5.2f} THz, "
      f"E_vac {vac.get_potential_energy() - 63/64*Ep:.2f} eV, MD fail step {fail}; CPU {time.process_time()-t0:.0f} s")
