"""Lab 10.6 (a): structure relaxation with MACE-MP-0 (small), end to end with ASE.

Part A  cubic SrTiO3 (5 atoms): free relaxation of a stretched, rattled start versus a
        symmetry-constrained relaxation; effect of the force tolerance fmax.
Part B  neutral Si monovacancy in 64- and 216-atom cells: unrelaxed and relaxed formation
        energy, relaxation pattern of the four neighbours.
Part C  ROI: "pre-relax with the cheap model, finish with the expensive one". CHGNet plays
        the expensive reference (a stand-in for DFT; no DFT code runs here). We count how many
        reference force calls are needed from the raw start and from the MACE-relaxed start.

Run:  $MLIP_PY_A (MACE/CHGNet environment) 106_relax.py [A|B|C]   (CPU ~5 min in total, 1 thread; parts A+B then C)
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from importlib import import_module
C = import_module('103_common'); C.reexec_for('mace')
import torch; torch.set_num_threads(1)
from ase import Atoms
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.constraints import FixSymmetry
from ase.spacegroup.symmetrize import check_symmetry

DATA = os.path.join(os.path.dirname(HERE), 'data')
from ase.io import read, write
PARTS = sys.argv[1] if len(sys.argv) > 1 else 'ABC'   # parts can be run separately (shared-memory machine)
mace, _, info_m = C.make_calc('mace')
print('model:', info_m['label'], info_m['n_params'])
JS = os.path.join(DATA, '106_relax.json')
out = json.load(open(JS)) if os.path.exists(JS) else {}


def relax(atoms, calc, fmax, cell=True, sym=False, maxsteps=400):
    atoms = atoms.copy(); atoms.calc = calc
    if sym:
        atoms.set_constraint(FixSymmetry(atoms))
    obj = FrechetCellFilter(atoms) if cell else atoms
    hist = []
    opt = BFGS(obj, logfile=None)
    def log():
        f = obj.get_forces(); hist.append(float(np.sqrt((f ** 2).sum(1)).max()))
    opt.attach(log, interval=1)
    t0 = time.process_time(); opt.run(fmax=fmax, steps=maxsteps)
    return atoms, opt.nsteps, time.process_time() - t0, hist


def srtio3(a):
    return Atoms('SrTiO3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
                 cell=np.eye(3) * a, pbc=True)


# ------------------------------------------------------------------ Part A
def partA():
    print('\n=== Part A: cubic SrTiO3, start a = 4.10 A (exp 3.905, PBE 3.944) ===')
    start = srtio3(4.10)
    rattled = start.copy(); rattled.rattle(stdev=0.03, seed=0)
    resA = {}
    for tag, at0, sym in [('sym-constrained', start, True), ('free, rattled', rattled, False)]:
        at, n, cpu, hist = relax(at0, mace, fmax=1e-3, sym=sym)
        e = at.get_potential_energy(); L = at.cell.cellpar()
        sg = check_symmetry(at, 1e-3).international
        print(f'  {tag:16s}: steps {n:3d}, CPU {cpu:5.2f} s, E = {e:.5f} eV, a,b,c = {L[0]:.4f} {L[1]:.4f} {L[2]:.4f} A,'
              f' angles {L[3]:.2f} {L[4]:.2f} {L[5]:.2f}, space group {sg}')
        resA[tag] = dict(steps=n, cpu=cpu, E=e, cellpar=L.tolist(), sg=sg, hist=hist)
        at.set_constraint(); write(os.path.join(DATA, f'106_sto_{tag[:4]}.xyz'), at)
    write(os.path.join(DATA, '106_sto_rattled_start.xyz'), rattled)
    dE = (resA['free, rattled']['E'] - resA['sym-constrained']['E']) * 1000
    print(f'  E(free) - E(cubic) = {dE:.2f} meV per 5-atom cell')
    at = read(os.path.join(DATA, '106_sto_free.xyz'))
    d = at.get_all_distances(mic=True)[1, 2:]
    print('  free: Ti-O distances =', ' '.join(f'{x:.4f}' for x in sorted(d)), 'A')
    acub = resA['sym-constrained']['cellpar'][0]
    print(f'  cubic a = {acub:.4f} A : {100*(acub/3.905-1):+.2f}% vs exp 3.905, {100*(acub/3.944-1):+.2f}% vs PBE(VASP) 3.944')
    print('  force tolerance (sym-constrained, from a = 4.10 A):')
    tol = {}
    for fm in (0.05, 0.01, 0.001, 0.0001):
        at, n, cpu, _ = relax(start, mace, fmax=fm, sym=True)
        s = at.get_stress()[:3].mean() * 160.21766  # GPa
        tol[str(fm)] = dict(steps=n, a=at.cell.cellpar()[0], stress_GPa=s)
        print(f'    fmax {fm:7.4f} eV/A: steps {n:3d}, a = {at.cell.cellpar()[0]:.5f} A, residual pressure {-s:+.4f} GPa')
    out['A'] = resA; out['A_tol'] = tol


# ------------------------------------------------------------------ Part B
def partB():
    print('\n=== Part B: neutral Si monovacancy (MACE-MP-0 small) ===')
    si0, nB, cpuB, _ = relax(bulk('Si', 'diamond', a=5.43, cubic=True), mace, fmax=1e-4, sym=True)
    a_si = si0.cell.cellpar()[0]; e_si = si0.get_potential_energy() / len(si0)
    print(f'  bulk Si: a = {a_si:.4f} A, E/atom = {e_si:.5f} eV')
    for rep in (2, 3):
        sc = si0.copy(); sc.set_constraint(); sc = sc.repeat((rep, rep, rep)); N = len(sc)
        vac = sc.copy(); del vac[0]
        vac.calc = mace; Eu = vac.get_potential_energy()
        # neighbours of the removed site (site 0 at the origin) in the unrelaxed cell
        dist0 = np.linalg.norm(((sc.get_scaled_positions()[1:] + 0.5) % 1 - 0.5) @ sc.cell, axis=1)
        nbr = np.argsort(dist0)[:4]
        pert = vac.copy(); rng = np.random.default_rng(1)
        pert.positions[nbr] += rng.normal(0, 0.02, (4, 3))  # break Td so that D2d can appear
        rel, n, cpu, _ = relax(pert, mace, fmax=0.005, cell=False)
        Er = rel.get_potential_energy()
        Ef_u = Eu - (N - 1) * e_si; Ef_r = Er - (N - 1) * e_si
        dd = rel.get_all_distances(mic=True)[np.ix_(nbr, nbr)]
        pairs = sorted(dd[np.triu_indices(4, 1)])
        disp = np.linalg.norm(rel.positions - vac.positions, axis=1)
        out.setdefault('B', {})[str(N)] = dict(Ef_unrel=Ef_u, Ef_rel=Ef_r, steps=n, cpu=cpu, nn_pairs=pairs, a_si=a_si)
        print(f'  N = {N:3d}: Ef(unrelaxed) = {Ef_u:.3f} eV, Ef(relaxed) = {Ef_r:.3f} eV, relaxation {Ef_r-Ef_u:+.3f} eV,'
              f' steps {n}, CPU {cpu:.1f} s')
        print(f'           neighbour-neighbour distances = {" ".join(f"{x:.3f}" for x in pairs)} A'
              f' (ideal {np.sqrt(2)*a_si/2:.3f}); max displacement {disp.max():.3f} A')
        if N == 64:
            write(os.path.join(DATA, '106_vac64_start.xyz'), pert); rel.set_constraint()
            write(os.path.join(DATA, '106_vac64_mace.xyz'), rel)
        json.dump(out, open(JS, 'w'), indent=1, default=float)


# ------------------------------------------------------------------ Part C
def partC():
    chg, _, info_c = C.make_calc('chgnet')
    print(f'\n=== Part C: pre-relax with MACE, finish with the reference ({info_c["label"]} as the stand-in for DFT) ===')
    cases = [('SrTiO3 free, rattled', '106_sto_rattled_start.xyz', '106_sto_free.xyz', dict(cell=True)),
             ('Si vacancy 63 atoms', '106_vac64_start.xyz', '106_vac64_mace.xyz', dict(cell=False))]
    resC = {}
    for tag, f_raw, f_pre, kw in cases:
        raw, pre = read(os.path.join(DATA, f_raw)), read(os.path.join(DATA, f_pre))
        a1, n1, c1, _ = relax(raw, chg, fmax=0.01, **kw)
        a2, n2, c2, _ = relax(pre, chg, fmax=0.01, **kw)
        _, nm, cm, _ = relax(raw, mace, fmax=0.01, **kw)
        e1, e2 = a1.get_potential_energy(), a2.get_potential_energy()
        resC[tag] = dict(ref_from_raw=n1 + 1, ref_from_mace=n2 + 1, mace_steps=nm + 1, dE_meV=(e2 - e1) * 1000)
        print(f'  {tag:22s}: reference calls from raw start {n1+1:3d} | MACE calls {nm+1:3d} then reference calls {n2+1:3d}'
              f' | final E difference {1000*(e2-e1):+.2f} meV')
    out['C'] = resC


for P, fn in (('A', partA), ('B', partB), ('C', partC)):
    if P in PARTS:
        fn(); json.dump(out, open(JS, 'w'), indent=1, default=float)
print(f'\nCPU total {time.process_time():.1f} s')
