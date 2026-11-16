"""Ex 10.3.6: float32 round-off floor of the total energy, NaCl 64-atom cell of Lab 10.5 Part F.
Run with /tmp/claude-0/v10/bin/python (MACE float64 and CHGNet float32)."""
import os, warnings, numpy as np, torch
warnings.filterwarnings('ignore'); torch.set_num_threads(1)
from ase.build import bulk
eps32 = float(np.finfo(np.float32).eps)
a = bulk('NaCl', 'rocksalt', a=5.64, cubic=True) * (2, 2, 2); a.rattle(0.1, seed=3)
from mace.calculators import MACECalculator
from chgnet.model import CHGNetCalculator
for name, calc in [('MACE-MP-0 small', MACECalculator(model_paths='../ch10/data/mace_mp0_small.model', device='cpu', default_dtype='float64')),
                   ('CHGNet 0.3.0', CHGNetCalculator(use_device='cpu', verbose=False))]:
    a.calc = calc; E = a.get_potential_energy(); F = a.get_forces()
    print(f"{name}: E = {E:.4f} eV ({E/len(a):.4f} eV/atom); RMS force {np.sqrt((F**2).mean()):.3f} eV/A; "
          f"eps32*|E| = {eps32*abs(E):.2e} eV total = {eps32*abs(E)/len(a):.2e} eV/atom; ulp(E) in float32 = {np.spacing(np.float32(E)):.2e} eV")
print(f"float32 eps = {eps32:.3e}")
# (d) random-walk estimate of energy error from float32 force noise in NVE (Si 64 atoms, 1200 K start, dt 2 fs)
kB = 8.617333e-5; m = 28.0855 * 1.66053907e-27; T = 467.0
v = np.sqrt(1.380649e-23 * T / m) * 1e-5           # A/fs per component
for dF in [1e-6, 1e-5]:
    for dt in [2.0, 1.0]:
        per_step = dF * v * dt * np.sqrt(3 * 64)      # eV, random sign per component
        for nsteps in [400, 1e6, 1e8]:
            print(f"dF={dF:.0e} eV/A dt={dt} fs: per-step work noise {per_step:.2e} eV; after {nsteps:.0e} steps "
                  f"sqrt(n) growth {per_step*np.sqrt(nsteps)/64*1e3:.2e} meV/atom; fully correlated {per_step/np.sqrt(192)*192*nsteps/64*1e3:.2e} meV/atom")
print(f"thermal speed per component at {T} K: {v:.4e} A/fs; position ulp at 10 A in float32 {np.spacing(np.float32(10.0)):.2e} A; step v*dt(2 fs) {2*v:.2e} A")
