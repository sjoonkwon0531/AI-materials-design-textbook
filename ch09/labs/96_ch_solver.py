"""Cahn-Hilliard solver of Ch.2 (Lab 2.14, Derivation Box 2.22), re-implemented in batched form.

    dc/dt = lap(c^3 - c) - eps^2 lap^2 c        on the periodic square (0, L)^2
    c_hat^{n+1} = [ (1 + dt S k^2) c_hat^n - dt k^2 FFT(c^3 - c)^n ] / (1 + dt S k^2 + dt eps^2 k^4)

Diagnostics use the same definitions as Lab 2.14:
    free energy  F = sum[ (c^2-1)^2/4 + eps^2 |grad c|^2 / 2 ] dx^2   (spectral gradient)
    length       L(t) = 2 pi / <k>,  <k> = sum(|k| S(k)) / sum(S(k)),  S(k) = |FFT(c - mean)|^2
Run as a script to reproduce the three structure-preservation numbers of Lab 2.14 (256^2).
"""
import time
import numpy as np
import scipy.fft as sfft


class CH2D:
    def __init__(self, N=64, L=2 * np.pi, eps=0.10, dt=0.05, S=2.0):
        self.N, self.L, self.eps, self.dt, self.S = N, L, eps, dt, S
        self.dx = L / N
        k1 = sfft.fftfreq(N, d=L / N) * 2 * np.pi
        kr = sfft.rfftfreq(N, d=L / N) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(k1, kr, indexing="ij")          # rfft layout
        self.K2 = self.KX**2 + self.KY**2
        self.kmag = np.sqrt(self.K2)
        self.gain = 1 + dt * S * self.K2
        self.denom = 1 + dt * S * self.K2 + dt * eps**2 * self.K2**2
        # weights that turn a sum over the half spectrum into a sum over the full spectrum
        self.wt = np.full(self.K2.shape, 2.0)
        self.wt[:, 0] = 1.0
        if N % 2 == 0:
            self.wt[:, -1] = 1.0

    def rfft(self, c):
        return sfft.rfft2(c, axes=(-2, -1))

    def irfft(self, ch):
        return sfft.irfft2(ch, s=(self.N, self.N), axes=(-2, -1))

    def advance(self, c, nsteps):
        """Advance a batch c (..., N, N) by nsteps solver steps (Box 2.22)."""
        ch = self.rfft(c)
        for _ in range(nsteps):
            ch = (self.gain * ch - self.dt * self.K2 * self.rfft(c**3 - c)) / self.denom
            c = self.irfft(ch)
        return c

    def trajectory(self, c0, n_rec, steps_per_rec):
        out = np.empty((c0.shape[0], n_rec + 1, self.N, self.N))
        out[:, 0] = c0
        c = c0
        for r in range(n_rec):
            c = self.advance(c, steps_per_rec)
            out[:, r + 1] = c
        return out

    def free_energy(self, c):
        ch = self.rfft(c)
        cx, cy = self.irfft(1j * self.KX * ch), self.irfft(1j * self.KY * ch)
        return np.sum(0.25 * (c**2 - 1) ** 2 + 0.5 * self.eps**2 * (cx**2 + cy**2), axis=(-2, -1)) * self.dx**2

    def length(self, c):
        Sk = np.abs(self.rfft(c - c.mean(axis=(-2, -1), keepdims=True))) ** 2 * self.wt
        return 2 * np.pi * Sk.sum((-2, -1)) / (self.kmag * Sk).sum((-2, -1))

    def structure_factor(self, c):
        """Radially averaged S(k) with bin width 2 pi / L; returns (k, S) averaged over the batch."""
        Sk = np.abs(self.rfft(c - c.mean(axis=(-2, -1), keepdims=True))) ** 2 * self.wt
        Sk = Sk.reshape(-1, *Sk.shape[-2:]).mean(0)
        dk = 2 * np.pi / self.L
        kb = np.rint(self.kmag / dk).astype(int).ravel()
        cnt = np.bincount(kb, weights=self.wt.ravel())
        prof = np.bincount(kb, weights=Sk.ravel()) / np.maximum(cnt, 1)
        return dk * np.arange(len(prof)), prof / self.N**4


def random_ic(rng, B, N, mean, amp=0.1):
    """Quench: uniform noise of amplitude amp around the given mean composition (scalar or (B,))."""
    mean = np.broadcast_to(np.asarray(mean, dtype=float), (B,))
    return mean[:, None, None] + amp * (2 * rng.random((B, N, N)) - 1)


if __name__ == "__main__":
    # Reproduce Lab 2.14: N=256, L=4 pi, eps=0.1, dt=0.05, S=2, seed=1, 8000 steps (T=400)
    N = 256
    sol = CH2D(N=N, L=4 * np.pi)
    c = 0.1 * (2 * np.random.default_rng(1).random((N, N)) - 1)
    mean0 = c.mean()
    t, F, Lt, drift = [0.0], [sol.free_energy(c)], [sol.length(c)], 0.0
    t0 = time.process_time()
    for n in range(1, 8001):
        c = sol.advance(c, 1)
        drift = max(drift, abs(c.mean() - mean0))
        if n % 20 == 0:
            t.append(n * sol.dt); F.append(sol.free_energy(c)); Lt.append(sol.length(c))
    el = time.process_time() - t0
    t, F, Lt = np.array(t), np.array(F), np.array(Lt)
    m = (t >= 50) & (t <= 400)
    slope = np.polyfit(np.log(t[m]), np.log(Lt[m]), 1)[0]
    print("Lab 2.14 re-run with the batched re-implementation (256^2, T = 400, seed 1)")
    print(f"[1] MASS conservation:   max |mean(t)-mean0| = {drift:.2e}")
    print(f"[2] FREE ENERGY:         F: {F[0]:.2f} -> {F[-1]:.2f},  F-increase steps = {(np.diff(F) > 0).sum()} / {len(F) - 1}")
    print(f"[3] COARSENING:          late-time log-log slope = {slope:.2f}")
    print(f"CPU time {el:.1f} s  ({1e3 * el / 8000:.2f} ms per step)")
