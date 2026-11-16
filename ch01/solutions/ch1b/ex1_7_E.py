"""Ex 1.7.E: I = int exp(-x^4) dx = 2 Gamma(5/4) by uniform MC, Gaussian IS, Gauss-Hermite."""
import numpy as np
from scipy.special import gamma
I = 2*gamma(1.25); print(f"exact I = 2*Gamma(5/4) = {I:.15f}")
rng = np.random.default_rng(2024); f = lambda x: np.exp(-x**4)
N = 10**5; R = 200                         # R independent repetitions to measure the spread
for L in [3.0, 5.0, 10.0]:
    est = np.array([2*L*np.mean(f(rng.uniform(-L, L, N))) for _ in range(R)])
    x = rng.uniform(-L, L, 10**6); sd1 = np.std(2*L*f(x))
    print(f"(a) uniform [-{L:g},{L:g}], N={N}: mean={est.mean():.6f}, rms err={np.sqrt(np.mean((est-I)**2)):.2e}, per-sample sd={sd1:.4f}")
p = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)
est = np.array([np.mean(f(z)/p(z)) for z in (rng.standard_normal(N) for _ in range(R))])
z = rng.standard_normal(10**6); sd2 = np.std(f(z)/p(z))
print(f"(b) IS N(0,1), N={N}: mean={est.mean():.6f}, rms err={np.sqrt(np.mean((est-I)**2)):.2e}, per-sample sd={sd2:.4f}")
for s in [0.5, 0.6, 0.7, 0.8, 1.0]:
    z = s*rng.standard_normal(10**6); q = np.exp(-z**2/(2*s*s))/(np.sqrt(2*np.pi)*s)
    print(f"    IS N(0,{s}^2): per-sample sd={np.std(f(z)/q):.4f}")
x = rng.uniform(-3, 3, 10**6); sd_u3 = np.std(6*f(x))
print(f"variance ratio uniform[-3,3]/IS N(0,1): {(sd_u3/sd2)**2:.2f}")
for n in [2, 4, 8, 16, 32, 64, 128]:
    xg, wg = np.polynomial.hermite.hermgauss(n)
    v = np.sum(wg*np.exp(-xg**4 + xg**2)); print(f"(c) Gauss-Hermite n={n:3d}: {v:.12f}, err={abs(v-I):.2e}")
# rescaled Gauss-Hermite: x = y/s with weight exp(-y^2)
for s, n in [(1.5, 16), (1.5, 32), (2.0, 32)]:
    yg, wg = np.polynomial.hermite.hermgauss(n); x = yg/s
    v = np.sum(wg*np.exp(-x**4 + yg**2))/s; print(f"    scaled GH (x=y/{s}) n={n}: err={abs(v-I):.2e}")
for n in [10, 20, 40]:
    xg, wg = np.polynomial.legendre.leggauss(n); L = 3.0
    v = L*np.sum(wg*f(L*xg)); print(f"    Gauss-Legendre on [-3,3] n={n}: err={abs(v-I):.2e}")
from scipy.special import gammaincc
print(f"tail mass |x|>3 = Gamma(1/4) Q(1/4,81)/2 = {gamma(0.25)*gammaincc(0.25,81.0)/2:.2e}")
