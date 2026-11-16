"""Lab 10.3 diagnostic: does the NVE energy drift shrink with the time step? (Si 64, same start as 103_bench nve)
If the drift comes from the integrator it falls roughly as dt^2; if it stays, it comes from the model
(non-smooth or noisy PES, finite precision). Usage: python 103_nve_dt.py <model>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('103_common'); model = sys.argv[1]; C.reexec_for(model)
import time, numpy as np
from ase.build import bulk
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
calc, net, info = C.make_calc(model); out = {}
for dt in (1.0, 2.0):
    a = bulk('Si', 'diamond', a=5.47, cubic=True) * (2, 2, 2); a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=1200, rng=np.random.default_rng(7)); Stationary(a)
    n = int(round(400 / dt)); E = []
    dyn = VelocityVerlet(a, dt * units.fs); dyn.attach(lambda: E.append(a.get_total_energy() / len(a)), interval=1)
    t0 = time.process_time(); dyn.run(n); cpu = time.process_time() - t0
    E = np.array(E); t = np.arange(len(E)) * dt / 1000; k = len(E) // 2
    slope = 1e3 * np.polyfit(t[k:], E[k:], 1)[0]
    out[str(dt)] = dict(drift=float(slope), std=float(1e3 * E[k:].std()), dE_end=float(1e3 * (E[-1] - E[0])))
    print(f"[{model}] dt = {dt} fs, {n} steps (0.4 ps): drift (2nd half) = {slope:+.3f} meV/atom/ps, "
          f"std = {out[str(dt)]['std']:.3f} meV/atom, E(end)-E(0) = {out[str(dt)]['dE_end']:+.3f} meV/atom, CPU {cpu:.0f} s")
C.save(model, 'nve_dt', out)
