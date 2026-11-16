"""Data for Lab 9.5 (b): 2D Navier-Stokes vorticity trajectories on the torus.
240 initial conditions from the Gaussian random field of Li et al. (seed 0), nu = 1e-3,
64 x 64 pseudo-spectral solver (CN + AB2, 2/3 dealiasing), dt = 1e-2, recorded every 1.0 up to T = 20.
Output: ../data/95_ns_nu1e-3_64.npz  (w: (240, 21, 64, 64) float32; first 200 train, last 40 test)."""
import time, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("ns", "95_ns_solver.py")
ns = importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)

N, NU, DT, T, NTRAJ, BATCH, SEED = 64, 1e-3, 1e-2, 20, 240, 40, 0
rng = np.random.default_rng(SEED)
solver = ns.NS2D(N, nu=NU, dt=DT)
w = np.empty((NTRAJ, T + 1, N, N), dtype=np.float32)
w0_all = np.empty((NTRAJ, N, N))
t0 = time.perf_counter(); c0 = time.process_time()
for b in range(0, NTRAJ, BATCH):
    w0 = ns.grf_sample(rng, BATCH, N)
    w0_all[b:b + BATCH] = w0
    w[b:b + BATCH] = solver.trajectory(w0, T, int(round(1 / DT)))
    print(f"  {b + BATCH:4d} / {NTRAJ} trajectories, {time.perf_counter() - t0:6.1f} s", flush=True)
el, cpu = time.perf_counter() - t0, time.process_time() - c0
np.savez_compressed("../data/95_ns_nu1e-3_64.npz", w=w, w0=w0_all, nu=NU, dt=DT, T=T, seed=SEED)
std_t = w.std(axis=(0, 2, 3))
pers = np.sqrt(((w[:, 1:] - w[:, :-1]) ** 2).sum((2, 3)) / (w[:, 1:] ** 2).sum((2, 3))).mean(0)
print(f"generated {NTRAJ} trajectories x {T} unit-time steps ({NTRAJ * T * int(round(1 / DT)):,d} solver steps) in {el:.1f} s wall, {cpu:.1f} s CPU"
      f"  = {1e3 * cpu / NTRAJ / T:.1f} ms CPU per sample per unit time")
print("std of w at t = 0, 5, 10, 20:", " ".join(f"{std_t[i]:.3f}" for i in (0, 5, 10, 20)))
print("persistence error ||w(t+1)-w(t)||/||w(t+1)|| at t = 0, 5, 10, 19:", " ".join(f"{pers[i]:.3f}" for i in (0, 5, 10, 19)))
