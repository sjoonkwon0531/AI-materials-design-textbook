"""104_pool.py -- reference ("DFT stand-in") data for the Section 10.4 labs.

Oracle = MACE-MP-0 small (float32).  System = diamond Si, 64-atom cubic cell at the oracle's
own lattice constant.  Produces
  data/104_pool.npz      oracle Langevin MD at 300, 900, 1500 K (200 steps x 2 fs each, every step kept)
  data/104_rattle.npz    16 rattled crystals (sigma 0.03-0.15 A), no MD needed
  data/104_test.npz      independent 1500 K trajectory (other seed), frames 50..150 every 5 -> 21 frames
  data/104_transfer_*.npz  strained cells (+-4 %), vacancy cell (63 atoms), 1800 K frames
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.eos import EquationOfState
from ase.build import bulk

t0 = time.process_time()
calc = C.oracle('float32')
calc64 = C.oracle('float64')
calls = 0

# lattice constant of the oracle (8-atom cell, float64)
V, E = [], []
for a in np.linspace(5.38, 5.56, 7):
    at = bulk('Si', 'diamond', a=a, cubic=True); at.calc = calc64
    V.append(at.get_volume()); E.append(at.get_potential_energy())
v0, e0, B = EquationOfState(V, E).fit()
print(f"oracle a0 = {v0 ** (1 / 3):.4f} A, B = {B / units.GPa:.1f} GPa, E/atom = {e0 / 8:.4f} eV")


def md(T, nsteps, seed, keep, atoms=None):
    global calls
    at = C.si64() if atoms is None else atoms
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=2 * T, rng=np.random.default_rng(seed)); Stationary(at)
    dyn = Langevin(at, 2.0 * units.fs, temperature_K=T, friction=0.02 / units.fs, rng=np.random.default_rng(seed + 100))
    out = []
    for s in range(1, nsteps + 1):
        dyn.run(1); calls += 1
        if keep(s):
            out.append(dict(pos=at.positions.copy(), cell=at.cell.array.copy(), Z=at.numbers.copy(),
                            E=float(at.get_potential_energy()), F=at.get_forces().copy(), T=T, step=s, tag=f'md{T}'))
    return out


pool = []
for T in (300, 900, 1500):
    pool += md(T, 200, seed=T, keep=lambda s: True)
C.save_frames(os.path.join(C.DATA, '104_pool.npz'), pool)
print(f"pool: {len(pool)} frames from oracle MD at 300/900/1500 K")

rng = np.random.default_rng(7)
rat = []
for k in range(16):
    at = C.si64(); sig = rng.uniform(0.03, 0.15)
    at.positions += rng.normal(0, sig, at.positions.shape)
    rat.append(dict(pos=at.positions.copy(), cell=at.cell.array.copy(), Z=at.numbers.copy(), tag=f'rattle{sig:.3f}'))
rat = C.label(calc, rat); calls += len(rat)
C.save_frames(os.path.join(C.DATA, '104_rattle.npz'), rat)

test = md(1500, 150, seed=4242, keep=lambda s: s >= 50 and s % 5 == 0)
C.save_frames(os.path.join(C.DATA, '104_test.npz'), test)
print(f"test: {len(test)} frames, independent 1500 K trajectory")

# transfer sets
st = []
for k, eps in enumerate([-0.04, -0.02, 0.02, 0.04, -0.04, 0.04]):
    at = C.si64(C.A0 * (1 + eps)); at.positions += rng.normal(0, 0.08, at.positions.shape)
    st.append(dict(pos=at.positions.copy(), cell=at.cell.array.copy(), Z=at.numbers.copy(), tag=f'strain{eps:+.2f}'))
st = C.label(calc, st); calls += len(st)
C.save_frames(os.path.join(C.DATA, '104_transfer_strain.npz'), st)
vac = C.si64(); del vac[0]
vac_fr = md(900, 60, seed=11, keep=lambda s: s % 10 == 0, atoms=vac)
C.save_frames(os.path.join(C.DATA, '104_transfer_vacancy.npz'), vac_fr)
hot = md(1800, 100, seed=18, keep=lambda s: s >= 40 and s % 10 == 0)
C.save_frames(os.path.join(C.DATA, '104_transfer_1800K.npz'), hot)

cpu = time.process_time() - t0
print(f"oracle calls: {calls}, CPU {cpu:.0f} s ({cpu / calls:.2f} s per 64-atom call incl. overhead)")
json.dump(dict(a0=v0 ** (1 / 3), B_GPa=B / units.GPa, calls=calls, cpu=cpu), open(os.path.join(C.DATA, '104_pool_meta.json'), 'w'))
