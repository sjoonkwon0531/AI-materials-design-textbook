"""Lab 10.3 diagnostic: Si C44 with clamped ions vs relaxed ions (internal relaxation, Kleinman zeta),
plus the energy of the relaxed sublattice shift. Strain gamma_yz = 1 %, model's own EOS a0."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('103_common'); model = sys.argv[1]; C.reexec_for(model)
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from ase import units
calc, net, info = C.make_calc(model)
d = json.load(open(os.path.join(C.DATA, f'103_{model}.json')))['eos']['Si']
a0 = (d['V0'] * 8) ** (1 / 3); base = bulk('Si', 'diamond', a=a0, cubic=True)
g = 0.01; out = {}
for relax in (False, True):
    s = []
    for sgn in (1, -1):
        e = np.zeros((3, 3)); e[1, 2] = e[2, 1] = sgn * g / 2
        a = base.copy(); a.set_cell(base.cell @ (np.eye(3) + e), scale_atoms=True); a.calc = calc
        x0 = a.positions.copy()
        if relax: BFGS(a, logfile=None).run(fmax=1e-4, steps=500)
        s.append(a.get_stress(voigt=True)[3] / units.GPa)
        if relax and sgn == 1: out['max_shift_A'] = float(np.abs(a.positions - x0).max())
    out['C44_relaxed' if relax else 'C44_clamped'] = float((s[0] - s[1]) / (2 * g))
print(f"[{model}] Si C44 clamped = {out['C44_clamped']:.1f} GPa, relaxed = {out['C44_relaxed']:.1f} GPa, "
      f"ratio {out['C44_relaxed']/out['C44_clamped']:.2f}, max ionic shift {out['max_shift_A']:.4f} A at 1 % shear")
C.save(model, 'c44', out)
