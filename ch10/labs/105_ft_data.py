"""105_ft_data.py -- labels from a *different reference* for the Section 10.5.3 fine-tuning lab.

Teacher (stand-in for "a higher level of theory than PBE"): GFN2-xTB via tblite, periodic, 8-atom
cubic diamond Si cells.  It is not more accurate than PBE; it is simply a different, physically
sensible potential-energy surface, so a fine-tuned model must move away from MACE-MP-0 to match it.
Configurations: isotropic strain -5..+5 %, small random shear, rattling, and frames of short
MACE-MP-0 MD at 600/1200 K (geometry only; labels always from xTB).
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')
from ase.build import bulk
from ase import units
from ase.eos import EquationOfState
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tblite.ase import TBLite

t0 = time.process_time()
def xtb():
    return TBLite(method='GFN2-xTB', verbosity=0, max_iterations=500)   # fresh per call: no stale SCF guess
mace = C.oracle('float32')


def configs(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        kind = k % 4
        a = bulk('Si', 'diamond', a=5.48, cubic=True)
        if kind == 0:                                   # isotropic strain + small rattle
            a.set_cell(a.cell * (1 + rng.uniform(-0.05, 0.05)), scale_atoms=True); a.rattle(0.03, seed=int(rng.integers(1e6)))
        elif kind == 1:                                 # random small shear/normal strain + rattle
            eps = rng.uniform(-0.02, 0.02, (3, 3)); eps = 0.5 * (eps + eps.T)
            a.set_cell(a.cell @ (np.eye(3) + eps), scale_atoms=True); a.rattle(0.05, seed=int(rng.integers(1e6)))
        elif kind == 2:                                 # larger rattle
            a.rattle(rng.uniform(0.08, 0.15), seed=int(rng.integers(1e6)))
        else:                                           # snapshot of a short MACE MD (geometry only)
            a.calc = mace; T = 600 if (k // 4) % 2 == 0 else 1200
            MaxwellBoltzmannDistribution(a, temperature_K=2 * T, rng=rng)
            Langevin(a, 2 * units.fs, temperature_K=T, friction=0.02 / units.fs, rng=rng).run(int(rng.integers(30, 80)))
            a.calc = None; a.set_velocities(np.zeros((8, 3)))
        out.append(a)
    return out


def labelled(atoms_list, n):
    out, fails = [], 0
    for a in atoms_list:
        a.calc = xtb()
        try:
            out.append(dict(pos=a.positions.copy(), cell=a.cell.array.copy(), Z=a.numbers.copy(),
                            E=float(a.get_potential_energy()), F=a.get_forces().copy()))
        except Exception:
            fails += 1                                 # SCF did not converge: skip this geometry
        if len(out) == n:
            break
    print(f"  {n} labelled, {fails} geometries skipped (SCF not converged)")
    return out


train = labelled(configs(1, 64), 48); test = labelled(configs(2, 24), 16)
C.save_frames(os.path.join(C.DATA, '105_xtb_train.npz'), train)
C.save_frames(os.path.join(C.DATA, '105_xtb_test.npz'), test)
V, E = [], []
for s in np.linspace(5.36, 5.66, 7):
    a = bulk('Si', 'diamond', a=s, cubic=True); a.calc = xtb()
    V.append(a.get_volume()); E.append(a.get_potential_energy())
v0, e0, B = EquationOfState(V, E).fit()
ref = dict(a0=v0 ** (1 / 3), B_GPa=B / units.GPa)
Fs = np.concatenate([f['F'].ravel() for f in train])
print(f"GFN2-xTB reference: a0 = {ref['a0']:.4f} A, B = {ref['B_GPa']:.1f} GPa")
print(f"labels: {len(train)} train + {len(test)} test 8-atom cells; force components std {Fs.std():.3f} eV/A")
cpu = time.process_time() - t0
print(f"xTB calls {len(train) + len(test) + 7}, CPU {cpu:.0f} s")
json.dump(dict(ref=ref, cpu=cpu), open(os.path.join(C.DATA, '105_xtb_meta.json'), 'w'))
