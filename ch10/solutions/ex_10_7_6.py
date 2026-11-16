"""Exercise 10.7.6: cost of MACE-MP-0 small MD on a 4x4x4 cubic SrTiO3 supercell (320 atoms); times
10 Langevin steps (2 fs) after a warm-up call, in float64 and float32.
Run: $MLIP_PY_A (MACE/CHGNet environment) ex_10_7_6.py"""
import time, numpy as np, torch, warnings
warnings.filterwarnings('ignore'); torch.set_num_threads(1)
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
a0 = 3.9433
u = Atoms('SrTiO3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], cell=np.eye(3) * a0, pbc=True)
for dtype in ('float64', 'float32'):
    at = u.repeat((4, 4, 4)); at.calc = MACECalculator(model_paths='../../ch10/data/mace_mp0_small.model', device='cpu', default_dtype=dtype)
    MaxwellBoltzmannDistribution(at, temperature_K=50, rng=np.random.default_rng(0)); at.get_forces()
    md = Langevin(at, 2 * units.fs, temperature_K=50, friction=0.01 / units.fs, rng=np.random.default_rng(1))
    t0 = time.process_time(); md.run(10); dt = (time.process_time() - t0) / 10
    print(f"{dtype}: {len(at)} atoms, CPU per MD step {dt:.2f} s -> 1,000 steps {dt*1000/60:.0f} min, 5 ps (2,500 steps) {dt*2500/3600:.1f} h")
