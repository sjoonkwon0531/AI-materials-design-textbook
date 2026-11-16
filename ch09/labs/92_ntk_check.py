"""Check of Derivation Box (NTK imbalance): for a model that is LINEAR in random Fourier features,
u(x) = sum_k a_k cos(2 pi b_k x) + c_k sin(2 pi b_k x),  b_k ~ N(0, sigma^2),
the residual of -u'' has features multiplied by (2 pi b_k)^2, so
(Tr K_rr / N_r) / (Tr K_uu / N_b) = mean_k (2 pi b_k)^4  ->  3 (2 pi sigma)^4  for many features."""
import numpy as np
PI = np.pi
xr = np.linspace(0, 1, 258)[1:-1]; xb = np.array([0.0, 1.0])
for sigma in (1.0, 10.0, 50.0):
    b = sigma * np.random.default_rng(0).standard_normal(4096)
    feat = lambda x: np.concatenate([np.cos(2 * PI * np.outer(x, b)), np.sin(2 * PI * np.outer(x, b))], 1)   # J_u = d u / d theta
    Ju = feat(xb); Jr = feat(xr) * np.tile((2 * PI * b) ** 2, 2)                                           # J_r = d(-u'') / d theta
    ratio = (np.sum(Jr**2) / len(xr)) / (np.sum(Ju**2) / len(xb))
    print(f"sigma = {sigma:4.0f}: linear RFF model ratio = {ratio:.3e},  mean (2 pi b)^4 = {np.mean((2*PI*b)**4):.3e},  3 (2 pi sigma)^4 = {3*(2*PI*sigma)**4:.3e}")
