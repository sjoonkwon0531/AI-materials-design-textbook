"""Ex 1.7.C: int_{|x|<=1} exp(-|x|^2) dx in d dims.
(a) tensor-product Gauss-Legendre on [-1,1]^d with the ball indicator (literal reading),
    computed exactly via the distribution of s = sum x_i^2 (integrand is radial);
(a') smooth reformulation: radial 1D Gauss  I = S_{d-1} int_0^1 r^{d-1} e^{-r^2} dr;
(b) Monte Carlo, uniform in cube with indicator, and uniform in ball."""
import numpy as np
from scipy.special import gammainc, gamma
def exact(d): return np.pi**(d/2)*gammainc(d/2, 1.0)
def tensor_gauss(d, n):
    x, w = np.polynomial.legendre.leggauss(n)
    s, W = np.array([0.0]), np.array([1.0])
    for _ in range(d):
        S = (s[:, None] + x[None, :]**2).ravel(); WW = (W[:, None]*w[None, :]).ravel()
        key = np.round(S, 12); u, inv = np.unique(key, return_inverse=True)
        s = u; W = np.bincount(inv, weights=WW)
    return np.sum(W*np.exp(-s)*(s <= 1.0))
def radial_gauss(d, n):
    x, w = np.polynomial.legendre.leggauss(n); r = 0.5*(x+1); w = 0.5*w
    Sd = 2*np.pi**(d/2)/gamma(d/2)
    if d == 1: return np.sum(w*np.exp(-r**2))*2
    return Sd*np.sum(w*r**(d-1)*np.exp(-r**2))
rng = np.random.default_rng(1)
eps = 1e-3   # target relative error
print(f"target relative error {eps:g}")
for d in [1, 2, 5, 10]:
    I = exact(d)
    # (a) smallest n (per dim) s.t. rel err <= eps, N = n^d
    nmax = {1: 400, 2: 400, 5: 60, 10: 24}[d]; hit = None; errs = []
    for n in range(2, nmax+1):
        e = abs(tensor_gauss(d, n)-I)/I; errs.append((n, e))
        if e <= eps and all(ee <= eps for _, ee in errs[-1:]): hit = n; break
    last = errs[-1]
    # (a') radial
    nr = next(n for n in range(1, 50) if abs(radial_gauss(d, n)-I)/I <= eps)
    # (b) MC: N needed = (sigma/(eps*I))^2, sigma from 10^6 samples
    M = 10**6
    xc = rng.uniform(-1, 1, (M, d)); s = (xc**2).sum(1)
    g = 2.0**d*np.exp(-s)*(s <= 1); Ncube = (g.std()/(eps*I))**2
    z = rng.standard_normal((M, d)); z /= np.linalg.norm(z, axis=1)[:, None]
    r = rng.uniform(0, 1, M)**(1/d); Vb = np.pi**(d/2)/gamma(d/2+1)
    gb = Vb*np.exp(-r**2); Nball = (gb.std()/(eps*I))**2
    print(f"d={d:2d}: I={I:.8f}, ball/cube vol={Vb/2**d:.3e}")
    print(f"   (a)  tensor Gauss+indicator: " + (f"n={hit}, N=n^d={float(hit)**d:.3e}" if hit else f"not reached up to n={last[0]} (N={float(last[0])**d:.2e}, rel err={last[1]:.1e})"))
    print(f"   (a') radial Gauss: n={nr} points")
    print(f"   (b)  MC cube+indicator: N~{Ncube:.3e};  MC uniform-in-ball: N~{Nball:.3e}")
