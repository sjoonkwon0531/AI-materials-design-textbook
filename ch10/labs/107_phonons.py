"""Lab 10.7: lattice dynamics of cubic SrTiO3 with four universal MLIPs (phonopy, 2x2x2 supercell).

For one model (argument: mace | chgnet | sevennet | mattersim) this script computes
  1. the cubic lattice constant and bulk modulus from an E(a) scan (Birch-Murnaghan fit);
  2. harmonic phonons by finite displacements (phonopy, 2x2x2 = 40 atoms, 0.01 A) at three
     lattice constants: the model's own a0, the PBE (VASP) value 3.944 A and the experimental
     3.905 A quoted by Evarestov et al., PRB 83, 134108 (2011); frequencies at G, X, M, R and a
     band path for the figure;
  3. the energy along a rigid a0a0c- (R-point) octahedral tilt in the 2x2x2 cell, and a full
     symmetry-constrained relaxation (I4/mcm, cell + positions) from a 4 degree start, which
     gives the tilt angle and the energy gain per formula unit;
  4. CPU time per 40-atom force call and for the whole phonon calculation.
Results go to data/107_<model>.json.

Run:  python 107_phonons.py mace      (re-executes itself under the right venv; ~1-3 min CPU per model)
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
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

DATA = os.path.join(os.path.dirname(HERE), 'data')
THZ2CM = 33.35641
calc, net, info = C.make_calc(model)
print(f'=== {info["label"]}  ({info["n_params"]:,} parameters, {info["dtype"]}) ===')
out = dict(info=info)


def sto(a, rep=1):
    at = Atoms('SrTiO3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
               cell=np.eye(3) * a, pbc=True)
    return at.repeat((rep, rep, rep)) if rep > 1 else at


def energy(at):
    a = at.copy(); a.calc = calc; return a.get_potential_energy()


# ---------------------------------------------------------------- 1. EOS
avals = np.linspace(3.86, 4.02, 9)
E = [energy(sto(a)) for a in avals]
eos = EquationOfState(avals ** 3, E, eos='birchmurnaghan'); v0, e0, B = eos.fit()
a0 = v0 ** (1 / 3); B_GPa = B * 160.21766
print(f'  EOS: a0 = {a0:.4f} A, B = {B_GPa:.1f} GPa   (exp a = 3.905 A; PBE/VASP 3.944 A)')
out['eos'] = dict(a0=a0, B=B_GPa)

# ---------------------------------------------------------------- 2. phonons
QP = {'G': [0, 0, 0], 'X': [0, .5, 0], 'M': [.5, .5, 0], 'R': [.5, .5, .5]}
PATH = [('G', 'X'), ('X', 'M'), ('M', 'G'), ('G', 'R'), ('R', 'X'), ('M', 'R')]


def phonons(a, band=False):
    u = sto(a)
    ph = Phonopy(PhonopyAtoms(symbols=u.get_chemical_symbols(), cell=u.cell[:], scaled_positions=u.get_scaled_positions()),
                 supercell_matrix=np.eye(3, dtype=int) * 2)
    ph.generate_displacements(distance=0.01)
    t0 = time.process_time(); forces = []
    for sc in ph.supercells_with_displacements:
        at = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True); at.calc = calc
        f = at.get_forces(); forces.append(f - f.mean(0))
    cpu = time.process_time() - t0
    ph.forces = forces; ph.produce_force_constants(); ph.symmetrize_force_constants()
    fr = {k: np.sort(ph.get_frequencies(q)) * THZ2CM for k, q in QP.items()}
    res = dict(a=a, ndisp=len(forces), cpu=cpu, freqs={k: v.tolist() for k, v in fr.items()})
    if band:
        segs = []
        for s, e in PATH:
            qs = np.linspace(QP[s], QP[e], 31); ph.run_qpoints(qs)
            segs.append((np.array(ph.get_qpoints_dict()['frequencies']) * THZ2CM).tolist())
        res['band'] = segs
    return res


def fmt(v):  # imaginary frequencies printed as e.g. 133i
    return ' '.join(f'{abs(x):.0f}i' if x < -0.5 else f'{x:.0f}' for x in v)


out['ph'] = {}
for tag, a in [('own', a0), ('pbe', 3.944), ('exp', 3.905)]:
    r = phonons(a, band=(tag in ('own', 'pbe')))
    out['ph'][tag] = r
    print(f'  phonons at a = {a:.4f} A ({tag}), {r["ndisp"]} displaced 40-atom supercells, CPU {r["cpu"]:.1f} s')
    for k in QP:
        print(f'     {k}: {fmt(r["freqs"][k][:6])}  ... max {r["freqs"][k][-1]:.0f} cm-1')

# ---------------------------------------------------------------- 3. R-point tilt
def tilted(a, phi_deg):
    at = sto(a, 2); u = 0.5 * a * np.tan(np.radians(phi_deg))
    for c in range(8):                       # 5 atoms per cubic cell, cells in repeat order
        i, j, k = c // 4, (c // 2) % 2, c % 2
        s = (-1) ** (i + j + k)
        at.positions[5 * c + 4, 1] += -s * u  # O at Ti - (a/2) x : moves along y
        at.positions[5 * c + 3, 0] += s * u   # O at Ti - (a/2) y : moves along x
    return at


def tilt_angle(at):
    """Average octahedral tilt from Ti-O-Ti angles of the equatorial O (phi = (180 - angle)/2)."""
    ang = []
    ti = [5 * c + 1 for c in range(8)]
    for c in range(8):
        for o in (5 * c + 3, 5 * c + 4):
            d = at.get_distances(o, ti, mic=True, vector=True); n = np.argsort(np.linalg.norm(d, axis=1))[:2]
            v1, v2 = d[n[0]], d[n[1]]
            ang.append(np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))))
    return (180 - np.mean(ang)) / 2


Ecub8 = energy(sto(a0, 2))
phis = np.arange(0, 10.01, 0.5)
prof = [(energy(tilted(a0, p)) - Ecub8) / 8 * 1000 for p in phis]
imin = int(np.argmin(prof))
print(f'  rigid a0a0c- tilt at a0: min {prof[imin]:.2f} meV/f.u. at {phis[imin]:.1f} deg;'
      f' E(10 deg) = {prof[-1]:+.1f} meV/f.u.')
st = tilted(a0, 4.0); st.calc = calc; st.set_constraint(FixSymmetry(st))
sg0 = check_symmetry(st, 1e-4).international
opt = BFGS(FrechetCellFilter(st), logfile=None); t0 = time.process_time(); opt.run(fmax=0.002, steps=300)
cpu_rel = time.process_time() - t0
phi_rel = tilt_angle(st); dE_rel = (st.get_potential_energy() - Ecub8) / 8 * 1000
cp = st.cell.cellpar()
print(f'  relaxed tilt ({sg0}, {opt.nsteps} steps, CPU {cpu_rel:.1f} s): phi = {phi_rel:.2f} deg,'
      f' dE = {dE_rel:.2f} meV/f.u., cell {cp[0]:.4f} {cp[1]:.4f} {cp[2]:.4f} A, c/a(pseudo) = {cp[2]/cp[0]:.5f}')
out['tilt'] = dict(phis=phis.tolist(), prof=prof, phi_rel=phi_rel, dE_rel=dE_rel, steps=opt.nsteps, sg=sg0,
                   cell=cp.tolist(), cpu=cpu_rel)

# ---------------------------------------------------------------- 4. timing
at = sto(a0, 2); at.rattle(0.01, seed=0); at.calc = calc
ts = []
for s in range(5):
    at.positions[0, 0] += 1e-4; t0 = time.process_time(); at.get_forces(); ts.append(time.process_time() - t0)
tcall = float(np.median(ts))
print(f'  CPU per 40-atom energy+force call (median of 5): {tcall*1000:.0f} ms;'
      f' total CPU of this script {time.process_time():.0f} s')
out['t_call'] = tcall; out['cpu_total'] = time.process_time()
json.dump(out, open(os.path.join(DATA, f'107_{model}.json'), 'w'), indent=1, default=float)
