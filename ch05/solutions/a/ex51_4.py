"""Ex 5.1.4: ideal (freely jointed) chain, N=100, b=1: check R* = sqrt(2N/3) b and <R^2> = N b^2."""
import numpy as np
rng = np.random.default_rng(0)
N, M = 100, 200000
R = np.zeros((M, 3))
for _ in range(N):
    v = rng.standard_normal((M, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    R += v
r = np.linalg.norm(R, axis=1)
print(f"<R^2> = {np.mean(r**2):.2f} (theory N b^2 = {N})")
h, e = np.histogram(r, bins=60, range=(0, 30), density=True)
c = 0.5 * (e[1:] + e[:-1])
# mode from a smooth fit of the Gaussian-chain form to the histogram
k = np.argmax(h)
print(f"histogram mode ~ {c[k]:.2f} (bin 0.5); theory R* = sqrt(2N/3) = {np.sqrt(2*N/3):.3f}")
# mean of W'(R)=0 estimate: fit ln P = A + 2 ln R - B R^2
m = h > 0
X = np.vstack([np.ones(m.sum()), np.log(c[m]), c[m]**2]).T
coef = np.linalg.lstsq(X, np.log(h[m]), rcond=None)[0]
print(f"fit ln P = A + p ln R - q R^2: p = {coef[1]:.3f} (theory 2), q = {-coef[2]:.5f} (theory 3/(2N) = {1.5/N:.5f})")
print(f"stiffness ratio 373.15/298.15 K = {373.15/298.15:.3f}")
