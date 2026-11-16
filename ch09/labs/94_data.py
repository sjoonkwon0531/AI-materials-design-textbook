"""§9.4 data: reference solutions of  u_t + c u_x + beta (u^2/2)_x = nu u_xx + rho (u - u^3)  (periodic).
Solver: ETDRK4, 256 points, dt = 5e-3, T = 1 (200 steps), 2/3 dealiasing; stored on 128 points, 41 snapshots.
Run:  python 94_data.py     (about 90 CPU-s)"""
import sys, time, importlib, numpy as np
sys.path.insert(0, "../../ch09/labs")
C = importlib.import_module("94_common")

# --- solver verification --------------------------------------------------------------------------
rng = np.random.default_rng(0)
n = 256; u0 = C.sample_ic(8, n, rng)
nu, c, beta, rho = C.sample_coeffs(8, rng, "pre")
k = np.fft.rfftfreq(n, 1 / n)
exact = np.fft.irfft(np.fft.rfft(u0) * np.exp(-nu[:, None] * k ** 2 - 1j * c[:, None] * k), n=n)
print(f"linear case (beta = rho = 0): max |u - exact| at T = 1: "
      f"{np.abs(C.etdrk4(u0, nu, c, 0 * nu, 0 * nu)[:, -1] - exact).max():.1e}")
for kind in ("pre", "react", "lownu"):
    nu, c, beta, rho = C.sample_coeffs(8, np.random.default_rng(1), kind)
    U = [C.etdrk4(u0, nu, c, beta, rho, dt=dt, n_save=4)[:, -1] for dt in (1e-2, 5e-3, 2.5e-3)]
    u0f = np.fft.irfft(np.fft.rfft(u0) * 2, n=512)                      # spectral interpolation to 512 points
    Uf = C.etdrk4(u0f, nu, c, beta, rho, dt=2.5e-3)[:, -1, ::2]
    e1, e2 = np.abs(U[0] - U[1]).max(), np.abs(U[1] - U[2]).max()
    print(f"{kind:6s}: |u(dt=1e-2) - u(5e-3)| = {e1:.1e}, |u(5e-3) - u(2.5e-3)| = {e2:.1e} "
          f"(ratio {e1 / e2:.1f}); |u_256 - u_512| = {np.abs(U[2] - Uf).max():.1e}")

# --- data sets ------------------------------------------------------------------------------------
sets = [("pre_train", 2000, 1, "pre", {}), ("pre_cal", 300, 2, "pre", {}), ("pre_test", 300, 3, "pre", {}),
        ("lownu_test", 300, 4, "lownu", {}), ("react_train", 256, 5, "react", {}),
        ("react_test", 300, 6, "react", {}), ("highk_test", 300, 7, "pre", dict(kmax=10, amp=1.5))]
tot_cpu, tot_n = 0.0, 0
for name, N, seed, kind, kw in sets:
    t0, w0 = time.process_time(), time.time()
    d = C.make_dataset(N, seed, kind, **kw)
    cpu, wall = time.process_time() - t0, time.time() - w0
    tot_cpu += cpu; tot_n += N
    np.savez(C.D + f"94_{name}.npz", **d)
    print(f"{name:12s} N = {N:4d}  kind = {kind:6s} {str(kw):24s} max|u| = {np.abs(d['u']).max():.2f}  "
          f"cpu {cpu:5.1f} s  wall {wall:5.1f} s")
print(f"solver cost: {1e3 * tot_cpu / tot_n:.1f} CPU-ms per trajectory (batched, 256 points, 200 steps); "
      f"total {tot_cpu:.0f} CPU-s for {tot_n} trajectories")
np.save(C.D + "94_solver_ms.npy", np.array([1e3 * tot_cpu / tot_n]))
