"""Reference data for Ch.7: GFN2-xTB (tblite) Langevin MD of ethanol.
Produces ethanol_xtb_{T}K.npz with R (n,9,3) Angstrom, Z (9,), E (n,) eV, F (n,9,3) eV/A.
Seeds fixed; dt = 0.5 fs; Langevin friction 0.01/fs; sample every 20 steps after 2,000-step warm-up."""
import numpy as np, sys, time
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from tblite.ase import TBLite

def run(T, nsamp, seed):
    a = molecule('CH3CH2OH')
    a.calc = TBLite(method="GFN2-xTB", verbosity=0)
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(a, temperature_K=T, rng=rng)
    dyn = Langevin(a, 0.5 * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=rng)
    dyn.run(2000)
    R, E, F = [], [], []
    for k in range(nsamp):
        dyn.run(20)
        R.append(a.get_positions().copy()); E.append(a.get_potential_energy()); F.append(a.get_forces().copy())
    np.savez(f"ethanol_xtb_{T}K.npz", R=np.array(R), Z=a.numbers, E=np.array(E), F=np.array(F), T=T, seed=seed)
    print(T, len(E), np.ptp(E), np.abs(np.array(F)).max())

t0 = time.time()
run(300, 3000, 1); run(600, 3000, 2); run(1000, 1000, 3)
print("time", time.time() - t0)
