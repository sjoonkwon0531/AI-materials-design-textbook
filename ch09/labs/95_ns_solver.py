"""Pseudo-spectral solver for the 2D Navier-Stokes equation in vorticity form on the
unit torus (0,1)^2,

    dw/dt + u . grad(w) = nu * lap(w) + f,   u = (psi_y, -psi_x),   lap(psi) = -w,

with the forcing and the Gaussian-random-field initial condition of Li et al. (FNO, ICLR 2021):
    f(x, y) = 0.1 * (sin(2 pi (x + y)) + cos(2 pi (x + y))),
    w0 ~ N(0, 7^{3/2} (-lap + 49 I)^{-2.5}).
Time stepping: Crank-Nicolson for the diffusion term, explicit second-order Adams-Bashforth
for advection (first step: forward Euler), 2/3-rule dealiasing of the quadratic term.
All arrays are batched: w has shape (B, N, N). float64 throughout.

Run as a script for a self-test (Taylor-Green decay, convergence in dt and N, timing).
"""
import numpy as np
import scipy.fft as sfft


def wavenumbers(N):
    k = sfft.fftfreq(N, d=1.0 / N)                 # integer wavenumbers 0..N/2-1, -N/2..-1
    kr = sfft.rfftfreq(N, d=1.0 / N)               # 0..N/2
    KX, KY = np.meshgrid(k, kr, indexing="ij")      # x: axis -2 (full), y: axis -1 (half)
    return KX, KY


def forcing(N):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 0.1 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))


def grf_sample(rng, B, N, alpha=2.5, tau=7.0):
    """Gaussian random field with covariance tau^{2 alpha - 2} (-lap + tau^2)^{-alpha} (zero mean),
    same construction as GaussianRF in the FNO data-generation code."""
    k = sfft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    sigma = tau ** (0.5 * (2 * alpha - 2))
    sqrt_eig = N**2 * np.sqrt(2.0) * sigma * (4 * np.pi**2 * (KX**2 + KY**2) + tau**2) ** (-alpha / 2)
    sqrt_eig[0, 0] = 0.0
    coeff = rng.standard_normal((B, N, N)) + 1j * rng.standard_normal((B, N, N))
    return np.real(sfft.ifft2(sqrt_eig * coeff, axes=(-2, -1)))


def spectral_resample(w, M):
    """Change resolution N -> M by truncating / zero-padding the Fourier coefficients."""
    N = w.shape[-1]
    if M == N:
        return w.copy()
    wh = sfft.fft2(w, axes=(-2, -1))
    out = np.zeros(w.shape[:-2] + (M, M), dtype=complex)
    m = min(N, M) // 2
    # keep modes |k| < m (drop the Nyquist mode of the smaller grid so that the result is real)
    idx_src = np.r_[0:m, N - m + 1:N]
    idx_dst = np.r_[0:m, M - m + 1:M]
    out[..., idx_dst[:, None], idx_dst[None, :]] = wh[..., idx_src[:, None], idx_src[None, :]]
    return np.real(sfft.ifft2(out, axes=(-2, -1))) * (M / N) ** 2


class NS2D:
    def __init__(self, N, nu=1e-3, dt=1e-3, force=True):
        self.N, self.nu, self.dt = N, nu, dt
        KX, KY = wavenumbers(N)
        self.ikx, self.iky = 2j * np.pi * KX, 2j * np.pi * KY
        lap = 4 * np.pi**2 * (KX**2 + KY**2)
        self.lap = lap
        self.inv_lap = np.where(lap > 0, 1.0 / np.where(lap > 0, lap, 1.0), 0.0)
        kmax = N // 2
        self.mask = ((np.abs(KX) <= (2.0 / 3.0) * kmax) & (np.abs(KY) <= (2.0 / 3.0) * kmax)).astype(float)
        self.fh = sfft.rfft2(forcing(N)) if force else 0.0
        self.cn_num = 1.0 - 0.5 * dt * nu * lap
        self.cn_den = 1.0 + 0.5 * dt * nu * lap

    def nonlinear(self, wh):
        """Fourier transform of -(u . grad w), dealiased."""
        psih = wh * self.inv_lap
        ir = lambda a: sfft.irfft2(a, s=(self.N, self.N), axes=(-2, -1))
        u, v = ir(self.iky * psih), ir(-self.ikx * psih)
        wx, wy = ir(self.ikx * wh), ir(self.iky * wh)
        return -self.mask * sfft.rfft2(u * wx + v * wy, axes=(-2, -1))

    def advance(self, w, nsteps):
        """Advance a batch w (B, N, N) by nsteps*dt. Restarts AB2 with one Euler step."""
        wh = sfft.rfft2(w, axes=(-2, -1))
        n_old = None
        for _ in range(nsteps):
            n_now = self.nonlinear(wh)
            adv = n_now if n_old is None else 1.5 * n_now - 0.5 * n_old
            wh = (self.cn_num * wh + self.dt * (adv + self.fh)) / self.cn_den
            n_old = n_now
        return sfft.irfft2(wh, s=(self.N, self.N), axes=(-2, -1))

    def trajectory(self, w0, n_rec, steps_per_rec):
        """Return array (B, n_rec + 1, N, N) with w at t = 0, steps_per_rec*dt, ...
        AB2 history is kept across records (only the very first step is Euler)."""
        wh = sfft.rfft2(w0, axes=(-2, -1))
        out = np.empty((w0.shape[0], n_rec + 1, self.N, self.N))
        out[:, 0] = w0
        n_old = None
        for r in range(n_rec):
            for _ in range(steps_per_rec):
                n_now = self.nonlinear(wh)
                adv = n_now if n_old is None else 1.5 * n_now - 0.5 * n_old
                wh = (self.cn_num * wh + self.dt * (adv + self.fh)) / self.cn_den
                n_old = n_now
            out[:, r + 1] = sfft.irfft2(wh, s=(self.N, self.N), axes=(-2, -1))
        return out


def rel_l2(a, b):
    """Relative L2 error per sample, ||a - b|| / ||b||, over the last two axes."""
    return np.sqrt(((a - b) ** 2).sum((-2, -1)) / (b**2).sum((-2, -1)))


if __name__ == "__main__":
    # [1] exact solution check: Taylor-Green vortex w = 2 k cos(kx) cos(ky) decays as exp(-2 nu k^2 t)
    N, nu = 64, 1e-2
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi
    w0 = (2 * k * np.cos(k * X) * np.cos(k * Y))[None]
    sol = NS2D(N, nu=nu, dt=1e-3, force=False)
    w1 = sol.advance(w0, 1000)
    exact = w0 * np.exp(-2 * nu * k**2 * 1.0)
    print(f"[1] Taylor-Green, N=64, nu=1e-2, t=1: rel. L2 error = {rel_l2(w1, exact)[0]:.2e}")

    # [2] temporal convergence at N = 64 (reference: dt = 2.5e-4) for 4 random initial conditions
    rng = np.random.default_rng(0)
    w0_128 = grf_sample(rng, 4, 128)
    w0_64 = spectral_resample(w0_128, 64)
    print(f"[2] GRF sample (4 fields): std(w0) = {w0_128.std():.3f}, max|w0| = {np.abs(w0_128).max():.3f}")
    ref_t = NS2D(64, dt=2.5e-4).advance(w0_64, 4 * 4000)          # t = 4
    prev = None
    for dt in [2e-2, 1e-2, 5e-3, 2.5e-3]:
        e = rel_l2(NS2D(64, dt=dt).advance(w0_64, int(round(4 / dt))), ref_t).mean()
        print(f"    N=64  dt={dt:.1e}: error vs dt=2.5e-4 at t=4 = {e:.2e}" + ("" if prev is None else f"   (ratio {prev / e:.2f})"))
        prev = e
    # [3] spatial convergence against N = 128 (dt = 1e-3), compared on the 64^2 grid
    ref_x = spectral_resample(NS2D(128, dt=1e-3).advance(w0_128, 4000), 64)
    for M, dt in [(64, 2.5e-3), (48, 5e-3), (32, 5e-3), (24, 1e-2), (16, 1e-2)]:
        w = NS2D(M, dt=dt).advance(spectral_resample(w0_128, M), int(round(4 / dt)))
        e = rel_l2(spectral_resample(w, 64), ref_x).mean()
        print(f"[3] N={M:<3d} dt={dt:.1e}: error vs N=128 at t=4 (on the 64^2 grid) = {e:.2e}")
    # [4] what the 2/3 rule does: same run without dealiasing
    sol = NS2D(64, dt=1e-2); raw = NS2D(64, dt=1e-2); raw.mask = np.ones_like(raw.mask)
    wa, wb = sol.advance(w0_64, 2000), raw.advance(w0_64, 2000)   # t = 20
    ref20 = spectral_resample(NS2D(128, dt=2.5e-3).advance(w0_128, 8000), 64)
    print(f"[4] N=64, t=20: error with 2/3 dealiasing = {rel_l2(wa, ref20).mean():.2e}, without = {rel_l2(wb, ref20).mean():.2e}")
    sol = NS2D(16, dt=2e-2); raw = NS2D(16, dt=2e-2); raw.mask = np.ones_like(raw.mask)
    w16 = spectral_resample(w0_128, 16)
    wa, wb = sol.advance(w16, 1000), raw.advance(w16, 1000)
    print(f"    N=16, t=20: error with 2/3 dealiasing = {rel_l2(spectral_resample(wa, 64), ref20).mean():.2e},"
          f" without = {rel_l2(spectral_resample(wb, 64), ref20).mean():.2e}")
