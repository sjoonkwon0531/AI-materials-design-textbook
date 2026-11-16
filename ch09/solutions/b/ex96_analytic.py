"""Exercises 9.6.1 - 9.6.3: closed-form numbers with numerical checks."""
import importlib.util, numpy as np
LABS = "../ch09/labs/"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ch = load("ch", LABS + "96_ch_solver.py")
eps = 0.1
print("=== 9.6.1 ===")
for m in (0.0, 0.3, 0.5):
    a = 1 - 3 * m * m
    print(f"(b) m = {m}: 1 - 3m^2 = {a:.2f}, k_c = {np.sqrt(a) / eps:.3f}, k* = {np.sqrt(a / 2) / eps:.3f}, sigma_max = {a * a / (4 * eps**2):.4f}, wavelength 2pi/k* = {2 * np.pi / (np.sqrt(a / 2) / eps):.3f}")
mc = np.sqrt((1 - 2 * eps**2) / 3); ms = 1 / np.sqrt(3)
print(f"(c) k* = 1: m = sqrt((1 - 2 eps^2)/3) = {mc:.5f}; spinodal 1/sqrt3 = {ms:.5f}; difference {ms - mc:.5f} ({100 * (ms - mc) / ms:.2f}%);"
      f" sigma_max there = {(1 - 3 * mc * mc)**2 / (4 * eps**2):.4f}; k_c = {np.sqrt(1 - 3 * mc * mc) / eps:.4f};  k_c < 1 (no unstable box mode) for m > {np.sqrt((1 - eps**2) / 3):.5f}")

print("=== 9.6.2 ===")
dt = 0.05
def g(k, m=0.0, S=2.0, dt=dt):
    return (1 + dt * k**2 * (S + 1 - 3 * m * m)) / (1 + dt * S * k**2 + dt * eps**2 * k**4)
sig = lambda k, m=0.0: k**2 * (1 - 3 * m * m - eps**2 * k**2)
for k in (4, 7):
    print(f"(c) k = {k}: numerator {1 + dt * k**2 * 3:.4f}, denominator {1 + dt * 2 * k**2 + dt * eps**2 * k**4:.4f}, g = {g(k):.5f}, g^20 = {g(k)**20:.2f};"
          f" sigma = {sig(k):.2f}, e^sigma = {np.exp(sig(k)):.3e};  ln(g)/dt = {np.log(g(k)) / dt:.3f}")
ks = np.arange(1, 33); print(f"    scheme: max g^20 = {(g(ks)**20).max():.2f} at k = {ks[np.argmax(g(ks))]};  PDE: max e^sigma = {np.exp(sig(ks)).max():.3e} at k = {ks[np.argmax(sig(ks))]}")
print(f"(d) S = 0: g(7) = {g(7, S=0):.5f}, g^20 = {g(7, S=0)**20:.1f}; ratio to S = 2: {g(7, S=0)**20 / g(7)**20:.0f};  ln g/dt: S=0 {np.log(g(7, S=0)) / dt:.2f}, S=2 {np.log(g(7)) / dt:.2f}, PDE {sig(7):.2f}")
print(f"    S = 0, k = 4: g^20 = {g(4, S=0)**20:.1f};  dt -> 0 check (dt = 1e-5): ln g/dt at k = 7: {np.log(g(7, dt=1e-5)) / 1e-5:.3f}")
# numerical check of (a): one scheme step applied to m + small mode
sol = ch.CH2D(64)
x = np.arange(64) * 2 * np.pi / 64; X, Y = np.meshgrid(x, x, indexing="ij")
for m, k in ((0.0, 4), (0.3, 7)):
    d = 1e-6; c1 = sol.advance(m + d * np.cos(k * X), 1)
    amp = 2 * np.abs(np.fft.fft2(c1 - m)[k, 0]) / 64**2
    print(f"(a) check, m = {m}, k = {k}: amplitude ratio after one solver step {amp / d:.6f}, formula g = {g(k, m):.6f}")

print("=== 9.6.3 ===")
L2 = (2 * np.pi) ** 2
for k in (32, 31, 16):
    dF = 0.02**2 * (0.5 + 0.25 * eps**2 * k**2)
    print(f"(c) k = {k}: per unit area {dF:.3e}; whole box x (2 pi)^2 = {dF * L2:.4f}  [bulk {0.02**2 * 0.5 * L2:.4f} + gradient {0.02**2 * 0.25 * eps**2 * k**2 * L2:.4f}]")
c0 = np.ones((64, 64))
for kx, ky in ((31, 0), (22, 22), (16, 0), (4, 0)):
    k = np.hypot(kx, ky); dlt = 0.02
    dF = sol.free_energy(c0 + dlt * np.cos(kx * X + ky * Y)) - sol.free_energy(c0)
    form = dlt**2 * (0.5 + 0.25 * eps**2 * k**2) * L2
    print(f"(b) check, k = ({kx},{ky}), |k| = {k:.2f}: solver free energy change {dF:.5f}; formula {form:.5f}; with the delta^4 term {form + 3 / 32 * dlt**4 * L2:.5f}")
d = np.load(LABS + "../data/96_ch_64.npz"); test = d["test"].astype(float)
F = np.stack([sol.free_energy(test[:, i]) for i in (18, 19, 20)], 1)
print(f"    solver, 40 test trajectories: F(45) {F[:, 0].mean():.4f}, F(47.5) {F[:, 1].mean():.4f}, F(50) {F[:, 2].mean():.4f}; mean drop 47.5 -> 50: {(F[:, 1] - F[:, 2]).mean():.4f}"
      f" (median {np.median(F[:, 1] - F[:, 2]):.4f}, min {(F[:, 1] - F[:, 2]).min():.4f}, max {(F[:, 1] - F[:, 2]).max():.4f})")
dstar = np.sqrt((F[:, 1] - F[:, 2]).mean() / ((0.5 + 0.25 * eps**2 * 32**2) * L2))
print(f"    amplitude whose energy equals the mean drop (k = 32): delta = {dstar:.4f}")
rng = np.random.default_rng(0); dF = []
for c in test[:, 20]:
    dF.append(sol.free_energy(c + 0.02 * np.cos(31 * X)) - sol.free_energy(c))
print(f"    adding 0.02 cos(31 x) to the solver states c(50): mean free-energy change {np.mean(dF):.4f}")
