"""Section 5.1 numbers.
(A) Estimate Box: how many MD steps (and how much wall time on this machine) to reach 1 ms with dt = 1 fs.
(B) Mori-Zwanzig in miniature: eliminate a fast variable y from a linear 2-variable system and
    compare the exact reduced equation (with memory) to the Markovian (memoryless) closure.
"""
import time, importlib, sys
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

sys.path.insert(0, "../../ch05/labs")
lj = importlib.import_module("52_ljmc")

# ---------------------------------------------------------------- (A)
steps = 1e-3 / 1e-15
print("=== (A) Estimate: 1 ms of MD at dt = 1 fs ===")
print(f"  steps needed = {steps:.0e}")
N, rho = 256, 0.6
L = (N / rho) ** (1 / 3)
pos = lj.fcc_lattice(N, L)
lj.forces_ts(pos, L)                       # compile
n = 2000
t = time.time()
for _ in range(n):
    lj.forces_ts(pos, L)
dt_force = (time.time() - t) / n
per_atom_step = dt_force / N
print(f"  this machine, N={N} LJ (numba, O(N^2) pair loop, 1 core): {dt_force*1e3:.3f} ms per force call "
      f"= {1/dt_force:.0f} steps/s")
print(f"  -> 1e12 steps would take {steps*dt_force/3.156e7:.0f} years for this 256-atom box")
# a production code with neighbour lists is O(N); use the measured per-atom cost at N=256 as an optimistic O(N) rate
for Nbig in [1e4, 1e6]:
    print(f"  optimistic O(N) extrapolation, N={Nbig:.0e}: {steps*per_atom_step*Nbig/3.156e7:.1e} years on one core")

# ---------------------------------------------------------------- (B)
print("=== (B) Mori-Zwanzig in miniature: dx/dt = -a x + b y, dy/dt = c x - g y; eliminate y ===")
a, b, c = 1.0, 1.0, -1.0
tt = np.linspace(0, 6, 601)
for g in [0.5, 2.0, 10.0, 50.0]:
    A = np.array([[-a, b], [c, -g]])
    x_exact = np.array([(expm(A * ti) @ np.array([1.0, 0.0]))[0] for ti in tt])
    lam = -a + b * c / g                    # Markovian closure: y slaved to x, y ~ (c/g) x
    x_mark = np.exp(lam * tt)
    err = np.max(np.abs(x_exact - x_mark))
    # exact reduced equation with memory kernel K(t) = b c exp(-g t) (y0 = 0 -> no noise term)
    def rhs(ti, u):                          # augment memory integral as an auxiliary ODE (exactly equivalent)
        x, I = u                             # I(t) = int_0^t exp(-g (t-s)) x(s) ds
        return [-a * x + b * c * I, x - g * I]
    sol = solve_ivp(rhs, (0, 6), [1.0, 0.0], t_eval=tt, rtol=1e-10, atol=1e-12)
    err_mem = np.max(np.abs(sol.y[0] - x_exact))
    print(f"  g/a = {g/a:5.1f} (fast/slow rate ratio): max|x_exact - x_Markov| = {err:.2e};  "
          f"with memory kernel: {err_mem:.1e}")

# ---------------------------------------------------------------- (C) rare events
print("=== (C) Rare event: harmonic-TST rate k = nu exp(-Ea/kT), nu = 1e13 1/s, T = 300 K ===")
kB = 8.617333262e-5
for Ea in [0.3, 0.5, 0.8]:
    k = 1e13 * np.exp(-Ea / (kB * 300))
    print(f"  Ea = {Ea:.1f} eV: Ea/kT = {Ea/(kB*300):.1f}, k = {k:.2e} 1/s, mean waiting time = {1/k:.2e} s"
          f" = {1/k/1e-15:.1e} MD steps of 1 fs")
