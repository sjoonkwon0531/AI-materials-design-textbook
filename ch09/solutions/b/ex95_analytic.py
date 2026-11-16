"""Exercises 9.5.1, 9.5.2, 9.5.4: closed-form numbers with numerical checks."""
import time, importlib.util, numpy as np
from scipy.linalg import solve_banded
LABS = "../ch09/labs/"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

print("=== 9.5.1 ===")
n = np.array([25, 50, 100, 200]); e = np.array([4.36e-4, 1.09e-4, 2.73e-5, 6.82e-6])
p = np.log2(e[:-1] / e[1:]); print("(a) p =", " ".join(f"{v:.3f}" for v in p))
slope, icpt = np.polyfit(np.log(n), np.log(e), 1); Cfit = np.exp(icpt)
print(f"    least-squares fit: p = {-slope:.3f}, C = {Cfit:.4f};  C from n = 200 with p = 2: {e[-1] * 200**2:.4f}")
n8 = 200 * np.sqrt(e[-1] / 1e-8); n8fit = (Cfit / 1e-8) ** (1 / -slope)
print(f"(b) n for 1e-8: from n = 200, p = 2: {n8:.0f};  from the fit: {n8fit:.0f}")
D = 0.1
def crank_nicolson(nx, nt):
    x = np.linspace(0, 1, nx + 1); dt = 1.0 / nt; r = D * dt / (1.0 / nx) ** 2
    u = np.sin(np.pi * x); m = nx - 1
    ab = np.zeros((3, m)); ab[0, 1:] = -r / 2; ab[1] = 1 + r; ab[2, :-1] = -r / 2
    for _ in range(nt):
        u = np.concatenate([[0.0], solve_banded((1, 1), ab, (1 - r) * u[1:-1] + (r / 2) * (u[2:] + u[:-2])), [0.0]])
    return x, u
for nn in (100, int(round(n8))):
    t0 = time.process_time(); x, u = crank_nicolson(nn, nn); cpu = time.process_time() - t0
    print(f"    Crank-Nicolson n = {nn}: max|err| at t=1 = {np.abs(u - np.exp(-D * np.pi**2) * np.sin(np.pi * x)).max():.3e}, CPU {cpu:.3f} s")
print(f"(c) (n/100)^2 = {(n8 / 100)**2:.0f};  1.80 ms x that = {1.80e-3 * (n8 / 100)**2:.2f} s;  from n = 800 (25.59 ms): {25.59e-3 * (n8 / 800)**2:.2f} s")
print(f"    measured CPU exponent between n = 100 and 800: {np.log(25.59 / 1.80) / np.log(8):.2f}")
print(f"    PINN: error ratio to 1e-8: soft {1.31e-4 / 1e-8:.0f}, hard {2.84e-5 / 1e-8:.0f}; loss would have to fall by ~{(2.84e-5 / 1e-8)**2:.1e} (hard: 4.7e-8 -> {4.7e-8 / (2.84e-5 / 1e-8)**2:.1e})")

print("=== 9.5.2 ===")
N = 64
print(f"(a) lowest aliased wavenumber N - 3K > K  <=>  K < N/4;  (b) N = 64: K < {N / 4:.0f}, keep |k| <= {N // 4 - 1} ({(N // 4 - 1) / (N / 2):.3f} of the band); 2/3 rule keeps |k| <= 21")
# brute-force check of (a): cube of a band-limited random function on N points, compare with the alias-free cube
rng = np.random.default_rng(0)
for K in (15, 16, 21):
    ch = np.zeros(N // 2 + 1, complex); ch[1:K + 1] = rng.standard_normal(K) + 1j * rng.standard_normal(K)
    c = np.fft.irfft(ch, n=N); cf = np.fft.irfft(ch * 4, n=4 * N)                 # same function on 4N points
    a = np.fft.rfft(c ** 3)[:K + 1]; b = np.fft.rfft(cf ** 3)[:K + 1] / 4
    print(f"    K = {K}: max aliasing error of (c^3)_k for |k| <= K: {np.abs(a - b).max() / np.abs(b).max():.1e}")
dt, S, eps = 0.05, 2.0, 0.1
for k in (4, 32, 64):
    den = 1 + dt * S * k**2 + dt * eps**2 * k**4
    print(f"(c) k = {k:2d}: denominator = 1 + {dt * S * k**2:.1f} + {dt * eps**2 * k**4:.1f} = {den:.1f};  dt k^2 / den = {dt * k**2 / den:.4f};  (1 + dt S k^2)/den = {(1 + dt * S * k**2) / den:.4f}")
print(f"    k -> inf: dt k^2/den -> 1/(eps^2 k^2) = {1 / (eps**2 * 64**2):.4f} at k = 64;  without the eps^2 k^4 term: (1 + dt S k^2)/den = 1")
ch = load("ch", LABS + "96_ch_solver.py")
sol = ch.CH2D(N=256, L=4 * np.pi, eps=0.1, dt=0.05, S=2.0)
c = 0.1 * (2 * np.random.default_rng(1).random((1, 256, 256)) - 1)
c = sol.advance(c, 200)                                                             # Lab 2.14 setting, t = 10
chh = np.fft.fft2(c[0]); M = 768; pad = np.zeros((M, M), complex)
idx = np.r_[0:128, M - 127:M]; src = np.r_[0:128, 129:256]
pad[np.ix_(idx, idx)] = chh[np.ix_(src, src)]; cf = np.real(np.fft.ifft2(pad)) * (M / 256) ** 2
exact = np.fft.fft2(cf ** 3)[np.ix_(idx, idx)] * (256 / M) ** 2; coarse = np.fft.fft2(c[0] ** 3)[np.ix_(src, src)]
kk = np.fft.fftfreq(256, d=4 * np.pi / 256)[src] * 2 * np.pi; KX, KY = np.meshgrid(kk, kk, indexing="ij"); kmag = np.sqrt(KX**2 + KY**2)
S2 = np.abs(chh[np.ix_(src, src)]) ** 2
print(f"    256^2, L = 4 pi at t = 10 (std c = {c.std():.2f}): share of |c_k|^2 at |k| > 32 (= N/4): {S2[kmag > 32].sum() / S2.sum():.1e}, at |k| > 43: {S2[kmag > 42.7].sum() / S2.sum():.1e}")
print(f"    aliasing error of FFT(c^3): ||coarse - exact|| / ||exact|| = {np.linalg.norm(coarse - exact) / np.linalg.norm(exact):.1e};"
      f" for |k| > 32: {np.linalg.norm((coarse - exact)[kmag > 32]) / np.linalg.norm(exact[kmag > 32]):.1e}")

print("=== 9.5.4 ===")
C, tf = 113 + 269, 3.39
for lab, ts in (("64^2", 18.19), ("32^2", 5.13), ("16^2", 1.95)):
    na = C / (20 * (ts - tf) * 1e-3); nb = C / (20 * (ts - tf / 50) * 1e-3)
    print(f"   {lab}: t_s = {ts} ms; (a) N* = {na:,.0f} (speed-up {ts / tf:.2f});  (b) t_FNO/50 = {tf / 50:.4f} ms: N* = {nb:,.0f} (speed-up {ts / (tf / 50):.0f})")
print(f"   lower bound C/(20 t_s): 64^2 {C / (20 * 18.19e-3):,.0f}, 32^2 {C / (20 * 5.13e-3):,.0f}, 16^2 {C / (20 * 1.95e-3):,.0f}")
print(f"   (b) fair: solver also 50x faster in batch -> 16^2: {1.95 / 50:.4f} ms vs FNO {tf / 50:.4f} ms; ratio unchanged {1.95 / tf:.2f}")
