"""Lab 10.7 check: are the soft modes robust to the finite-displacement amplitude?

Phonons of cubic SrTiO3 at a = 3.944 A (2x2x2) with displacements 0.005, 0.01, 0.02, 0.04 A.
float32 models (CHGNet, SevenNet, MatterSim) suffer from round-off at small amplitudes; large
amplitudes pick up anharmonicity. Run:  python3 107_check_disp.py <model>
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from importlib import import_module
C = import_module('103_common')
model = sys.argv[1]; C.reexec_for(model)
import torch; torch.set_num_threads(1)
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
calc, net, info = C.make_calc(model)
QP = {'G': [0, 0, 0], 'X': [0, .5, 0], 'M': [.5, .5, 0], 'R': [.5, .5, .5]}
a = 3.944
u = Atoms('SrTiO3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
          cell=np.eye(3) * a, pbc=True)
res = {}
print(f'=== {info["label"]} ({info["dtype"]}), a = {a} A: lowest frequency (cm-1) at G, X, M, R ===')
for d in (0.005, 0.01, 0.02, 0.04):
    ph = Phonopy(PhonopyAtoms(symbols=u.get_chemical_symbols(), cell=u.cell[:], scaled_positions=u.get_scaled_positions()),
                 supercell_matrix=np.eye(3, dtype=int) * 2)
    ph.generate_displacements(distance=d)
    F = []
    for sc in ph.supercells_with_displacements:
        at = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True); at.calc = calc
        f = at.get_forces(); F.append(f - f.mean(0))
    ph.forces = F; ph.produce_force_constants(); ph.symmetrize_force_constants()
    low = {k: float(np.sort(ph.get_frequencies(q))[0] * 33.35641) for k, q in QP.items()}
    acc = float(np.sort(np.abs(ph.get_frequencies([0, 0, 0])))[:3].max() * 33.35641)
    res[d] = low
    print(f'  disp {d:5.3f} A: ' + '  '.join(f'{k} {abs(v):5.1f}{"i" if v < 0 else " "}' for k, v in low.items())
          + f'   |acoustic at G| max {acc:.2f}')
json.dump(res, open(os.path.join(os.path.dirname(HERE), 'data', f'107_disp_{model}.json'), 'w'), indent=1)
