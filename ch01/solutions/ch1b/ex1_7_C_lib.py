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
