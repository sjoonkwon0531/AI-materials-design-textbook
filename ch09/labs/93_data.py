"""Sec.9.3 data: classical pseudo-spectral solvers that generate every training set of this section.

  burgers_solve : u_t + (u^2/2)_x = nu u_xx      on x in [0,1), periodic
  ks_solve      : u_t + (u^2/2)_x + u_xx + u_xxxx = 0  on x in [0,L), periodic
Both: Fourier collocation, 2/3-rule dealiasing, linear term treated exactly by an
integrating factor, classical RK4 on the nonlinear term (IFRK4).  Batched over initial conditions.

Run as a script to (1) validate against the Cole-Hopf exact solution and by step-halving,
(2) write data/93_burgers.npz and data/93_ks.npz.
Import (file name starts with a digit):  D = importlib.import_module("93_data")
"""
import os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def _ifrk4(u0, lin, nonlin_hat, dt, nsteps, save_every=None):
    """u0: (B,N) real.  lin: (Nf,) real symbol of the linear operator (u_hat' = lin*u_hat + N_hat)."""
    N = u0.shape[-1]
    uh = np.fft.rfft(u0, axis=-1)
    E, E2 = np.exp(lin * dt), np.exp(lin * dt / 2)
    out = [u0.copy()] if save_every else None
    for n in range(nsteps):
        k1 = nonlin_hat(uh)
        k2 = nonlin_hat(E2 * (uh + 0.5 * dt * k1))
        k3 = nonlin_hat(E2 * uh + 0.5 * dt * k2)
        k4 = nonlin_hat(E * uh + dt * E2 * k3)
        uh = E * uh + dt / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
        if save_every and (n + 1) % save_every == 0:
            out.append(np.fft.irfft(uh, n=N, axis=-1))
    return np.stack(out, 1) if save_every else np.fft.irfft(uh, n=N, axis=-1)


def _conv_flux(N, k):
    """returns nonlin_hat(uh) = -(ik/2) FFT[u^2] with 2/3 dealiasing."""
    mask = (np.arange(N // 2 + 1) <= N // 3).astype(float)
    def f(uh):
        u = np.fft.irfft(uh * mask, n=N, axis=-1)
        return -0.5j * k * np.fft.rfft(u * u, axis=-1) * mask
    return f


def burgers_solve(u0, nu, T, dt, save_every=None):
    N = u0.shape[-1]
    k = 2 * np.pi * np.arange(N // 2 + 1)                    # domain length 1
    return _ifrk4(u0, -nu * k**2, _conv_flux(N, k), dt, int(round(T / dt)), save_every)


def ks_solve(u0, L, T, dt, save_every=None):
    N = u0.shape[-1]
    k = 2 * np.pi / L * np.arange(N // 2 + 1)
    return _ifrk4(u0, k**2 - k**4, _conv_flux(N, k), dt, int(round(T / dt)), save_every)


def grf(n, N, rng, sigma=25.0, tau=5.0, alpha=2.0, kmax=None):
    """Gaussian random field on [0,1) with the spectrum of N(0, sigma^2 (-Lap + tau^2)^(-alpha)) (Li et al. 2021, Burgers):
    lam_k = sigma^2 / ((2 pi k)^2 + tau^2)^alpha,
    u = sqrt(lam_0) xi_0 + sum_{k>=1} 2 sqrt(lam_k) (xi_k cos 2 pi k x - eta_k sin 2 pi k x),  xi, eta ~ N(0,1).
    NOTE: the k>=1 amplitudes are sqrt(2) times those of the exact measure (whose coefficient on the orthonormal
    sqrt(2) cos, sqrt(2) sin is sqrt(lam_k)); the text states this."""
    kk = np.arange(N // 2 + 1)
    lam = sigma**2 / ((2 * np.pi * kk) ** 2 + tau**2) ** alpha
    if kmax is not None: lam = lam * (kk <= kmax)
    c = (rng.standard_normal((n, N // 2 + 1)) + 1j * rng.standard_normal((n, N // 2 + 1))) * np.sqrt(lam / 2)
    c[:, 0] = rng.standard_normal(n) * np.sqrt(lam[0])       # real zero mode
    c[:, 1:] *= np.sqrt(2.0)                                 # Re c_k, Im c_k ~ N(0, lam_k)
    c[:, -1] = 0.0
    return np.fft.irfft(c, n=N, axis=-1) * N


def cole_hopf_exact(x, t, nu, a=1.5):
    """phi = a + exp(-4 pi^2 nu t) cos(2 pi x) solves phi_t = nu phi_xx;  u = -2 nu phi_x / phi solves Burgers."""
    e = np.exp(-4 * np.pi**2 * nu * t)
    return 4 * np.pi * nu * e * np.sin(2 * np.pi * x) / (a + e * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    # ---------------- (1) validation ----------------
    nu = 0.005
    for N in (64, 128):
        x = np.arange(N) / N
        u = burgers_solve(cole_hopf_exact(x, 0.0, nu)[None], nu, 1.0, 5e-4)[0]
        print(f"Cole-Hopf check  N={N:4d} dt=5e-4 T=1: max|u-u_exact| = {np.abs(u - cole_hopf_exact(x, 1.0, nu)).max():.2e}")
    rng = np.random.default_rng(0)
    u0 = grf(32, 512, rng)                                   # same band-limited ICs on two grids
    u0h = np.fft.rfft(u0, axis=1); u0h[:, -1] = 0
    u0 = np.fft.irfft(u0h, n=512, axis=1)
    u0f = np.fft.irfft(np.concatenate([u0h, np.zeros((32, 256))], 1), n=1024, axis=1) * 2
    ref = burgers_solve(u0f, nu, 0.25, 1.25e-4)[:, ::2]       # N=1024, dt 4x smaller
    for dt in (1e-3, 5e-4):
        e = np.linalg.norm(burgers_solve(u0, nu, 0.25, dt) - ref, axis=1) / np.linalg.norm(ref, axis=1)
        print(f"Burgers GRF N=512 dt={dt:.1e} vs (N=1024, dt=1.25e-4): rel-L2 mean {e.mean():.1e}  max {e.max():.1e}")

    # ---------------- (2) Burgers data set ----------------
    t0 = time.time()
    NF, NU, T, DT = 512, 0.005, 0.25, 5e-4
    rng = np.random.default_rng(93)
    ntr, nte = 2000, 200
    a = grf(ntr + nte, NF, rng)
    traj = burgers_solve(a, NU, T, DT, save_every=50)        # snapshots every 0.025 -> (n, 11, 512)
    print(f"Burgers: {ntr}+{nte} ICs, N={NF}, nu={NU}, T={T}, dt={DT}: {time.time()-t0:.0f} s; "
          f"mean drift max {np.abs(traj[:, -1].mean(1) - a.mean(1)).max():.1e}")
    print(f"  u0: mean of |mean| {np.abs(a.mean(1)).mean():.3f}, rms fluctuation {a.std(1).mean():.3f}; "
          f"u(T): rms fluctuation {traj[:, -1].std(1).mean():.3f}")
    uh = np.abs(np.fft.rfft(traj[:, -1], axis=1)) ** 2
    tail = lambda kc: np.sqrt(uh[:, kc + 1:].sum(1) / uh.sum(1)).mean()
    print("  u(T): relative L2 content above k = 16, 32, 64, 128: %.2e  %.2e  %.2e  %.2e" % tuple(tail(kc) for kc in (16, 32, 64, 128)))
    np.savez_compressed(os.path.join(DATA, "93_burgers.npz"), traj=traj.astype(np.float32), nu=NU, T=T, ntr=ntr)

    # ---------------- (3) Kuramoto-Sivashinsky data set ----------------
    t0 = time.time()
    L, N, DT, DTS = 64.0, 128, 0.025, 1.0                   # solver step 0.025, stored step 1.0
    rng = np.random.default_rng(94)
    ntr, nte, burn, ns_tr, ns_te = 64, 16, 200.0, 200, 400
    x = np.arange(N) * L / N
    u0 = sum(rng.uniform(-0.5, 0.5, (ntr + nte, 1)) * np.cos(2 * np.pi * m * x / L + rng.uniform(0, 2 * np.pi, (ntr + nte, 1)))
             for m in range(1, 9))
    u0 = ks_solve(u0, L, burn, DT)                           # onto the attractor
    tr = ks_solve(u0, L, ns_te * DTS, DT, save_every=int(round(DTS / DT)))     # (n, 401, 128)
    chk0 = ks_solve(u0[:4], L, 20.0, DT); chk1 = ks_solve(u0[:4], L, 20.0, DT / 2)
    print(f"KS: {ntr}+{nte} trajectories, L={L}, N={N}, dt={DT}, stored dt={DTS}, {ns_te} steps: {time.time()-t0:.0f} s")
    print(f"  step-halving check over t=20: rel-L2 {np.linalg.norm(chk0-chk1)/np.linalg.norm(chk1):.1e}; "
          f"mean drift max {np.abs(tr.mean(2)).max():.1e}; rms u {tr.std():.3f}")
    np.savez_compressed(os.path.join(DATA, "93_ks.npz"), traj=tr.astype(np.float32), L=L, dts=DTS, ntr=ntr, ns_tr=ns_tr)
