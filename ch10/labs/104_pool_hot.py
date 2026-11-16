"""104_pool_hot.py -- extra oracle data for the hot (near-melting and liquid) regime of Lab 104_al.

Oracle = MACE-MP-0 small (float32).  64-atom Si, Langevin, 2 fs, friction 0.02/fs.
  data/104_pool_hot.npz   oracle MD at 2000 and 2500 K, 200 steps each, every step kept
  data/104_test_2500K.npz independent 2500 K trajectory, frames 50..150 every 5 (21 frames)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

t0 = time.process_time()
calc = C.oracle('float32')


def md(T, nsteps, seed, keep):
    at = C.si64(); at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=2 * T, rng=np.random.default_rng(seed)); Stationary(at)
    dyn = Langevin(at, 2.0 * units.fs, temperature_K=T, friction=0.02 / units.fs, rng=np.random.default_rng(seed + 100))
    out = []
    for s in range(1, nsteps + 1):
        dyn.run(1)
        if keep(s):
            out.append(dict(pos=at.positions.copy(), cell=at.cell.array.copy(), Z=at.numbers.copy(),
                            E=float(at.get_potential_energy()), F=at.get_forces().copy(), T=T, step=s, tag=f'md{T}'))
    dmin = neighbor_list('d', at, 3.0).min()
    cn = np.bincount(neighbor_list('i', at, 2.75), minlength=64).mean()
    print(f"  oracle {T} K, {nsteps} steps: final T {at.get_temperature():.0f} K, min Si-Si {dmin:.2f} A, mean CN(2.75 A) {cn:.2f}")
    return out


hot = md(2000, 200, 2000, lambda s: True) + md(2500, 200, 2500, lambda s: True)
C.save_frames(os.path.join(C.DATA, '104_pool_hot.npz'), hot)
test = md(2500, 150, 5252, lambda s: s >= 50 and s % 5 == 0)
C.save_frames(os.path.join(C.DATA, '104_test_2500K.npz'), test)
F = np.array([f['F'] for f in test])
print(f"hot pool {len(hot)} frames, 2500 K test {len(test)} frames, RMS force component {1000 * np.sqrt((F ** 2).mean()):.0f} meV/A")
print(f"oracle calls 550, CPU {time.process_time() - t0:.0f} s")
