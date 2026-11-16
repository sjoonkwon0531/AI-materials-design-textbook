"""Exercise 10.7.5: cubic CsPbBr3 with four universal MLIPs (same recipe as labs/107_phonons.py).
Cs at the corner, Pb at the body centre, Br at the face centres. For one model:
  1. E(a) on 9 points, 5.7-6.1 A, Birch-Murnaghan a0 and B;
  2. phonopy 2x2x2 (40 atoms), 0.01 A, at the model's own a0: lowest modes at G, X, M, R;
  3. rigid a0a0c- tilt profile 0-14 deg at a0, and an I4/mcm (FixSymmetry) relaxation of cell + positions
     from a 6 degree start.
Run: python ex_10_7_5.py <mace|chgnet|sevennet|mattersim>   (re-executes itself under the right venv)"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, '../ch10/labs')
from importlib import import_module
C = import_module('103_common')
model = sys.argv[1]; C.reexec_for(model)
import torch; torch.set_num_threads(1)
from ase import Atoms
from ase.eos import EquationOfState
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.constraints import FixSymmetry
from ase.spacegroup.symmetrize import check_symmetry
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

THZ2CM = 33.35641
calc, net, info = C.make_calc(model)
t00 = time.process_time()


def cpb(a, rep=1):
    at = Atoms('CsPbBr3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
               cell=np.eye(3) * a, pbc=True)
    return at.repeat((rep, rep, rep)) if rep > 1 else at


def energy(at):
    a = at.copy(); a.calc = calc; return a.get_potential_energy()


avals = np.linspace(5.7, 6.1, 9)
E = [energy(cpb(a)) for a in avals]
v0, e0, B = EquationOfState(avals ** 3, E, eos='birchmurnaghan').fit()
a0 = v0 ** (1 / 3); B_GPa = B * 160.21766
iE = int(np.argmin(E))

QP = {'G': [0, 0, 0], 'X': [0, .5, 0], 'M': [.5, .5, 0], 'R': [.5, .5, .5]}
u = cpb(a0)
ph = Phonopy(PhonopyAtoms(symbols=u.get_chemical_symbols(), cell=u.cell[:], scaled_positions=u.get_scaled_positions()),
             supercell_matrix=np.eye(3, dtype=int) * 2)
ph.generate_displacements(distance=0.01)
forces = []
for sc in ph.supercells_with_displacements:
    at = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True); at.calc = calc
    f = at.get_forces(); forces.append(f - f.mean(0))
ph.forces = forces; ph.produce_force_constants(); ph.symmetrize_force_constants()
fr = {k: np.sort(ph.get_frequencies(q)) * THZ2CM for k, q in QP.items()}


def lowval(k):  # at G drop the three acoustic modes (the three of smallest |omega|), elsewhere the lowest mode
    v = fr[k]
    if k == 'G':
        v = np.sort(np.delete(v, np.argsort(np.abs(v))[:3]))
    return float(v[0])


def low(k):
    v = lowval(k)
    return f'{abs(v):.0f}i' if v < -0.5 else f'{v:.0f}'


def tilted(a, phi_deg):
    at = cpb(a, 2); d = 0.5 * a * np.tan(np.radians(phi_deg))
    for c in range(8):
        i, j, k = c // 4, (c // 2) % 2, c % 2
        s = (-1) ** (i + j + k)
        at.positions[5 * c + 4, 1] += -s * d
        at.positions[5 * c + 3, 0] += s * d
    return at


def tilt_angle(at):
    ang = []; pb = [5 * c + 1 for c in range(8)]
    for c in range(8):
        for o in (5 * c + 3, 5 * c + 4):
            dv = at.get_distances(o, pb, mic=True, vector=True); n = np.argsort(np.linalg.norm(dv, axis=1))[:2]
            v1, v2 = dv[n[0]], dv[n[1]]
            ang.append(np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))))
    return (180 - np.mean(ang)) / 2


Ecub8 = energy(cpb(a0, 2))
phis = np.arange(0, 20.01, 1.0)
prof = np.array([(energy(tilted(a0, p)) - Ecub8) / 8 * 1000 for p in phis])
im = int(np.argmin(prof))
st = tilted(a0, 6.0); st.calc = calc; st.set_constraint(FixSymmetry(st))
sg = check_symmetry(st, 1e-4).international
opt = BFGS(FrechetCellFilter(st), logfile=None); opt.run(fmax=0.002, steps=300)
phi_rel = tilt_angle(st); dE_rel = (st.get_potential_energy() - Ecub8) / 8 * 1000; cp = st.cell.cellpar()
res = dict(model=model, a0=a0, B=B_GPa, edge=bool(iE in (0, len(avals) - 1)),
           low={k: lowval(k) for k in QP}, acoustic_G=sorted(np.abs(fr['G']))[:3],
           rigid_min=float(prof[im]), rigid_phi=float(phis[im]), phi_rel=float(phi_rel), dE_rel=float(dE_rel),
           ca=float(cp[2] / cp[0]), steps=opt.nsteps, sg=sg, cpu=time.process_time() - t00)
print(f"{info['label']:20s} a0 {a0:.4f} A, B {B_GPa:5.1f} GPa{' (min at scan edge!)' if res['edge'] else ''} | "
      f"lowest G {low('G')}, X {low('X')}, M {low('M')}, R {low('R')} cm-1 | rigid tilt min {prof[im]:.1f} meV/f.u. at {phis[im]:.0f} deg |"
      f" {sg} relax: {phi_rel:.2f} deg, {dE_rel:.1f} meV/f.u., c/a {cp[2]/cp[0]:.4f}, {opt.nsteps} steps | CPU {res['cpu']:.0f} s")
json.dump(res, open(f'../ch10/solutions/ex_10_7_5_{model}.json', 'w'), indent=1)
