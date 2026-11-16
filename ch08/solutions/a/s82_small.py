import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
rng = np.random.default_rng(0)
print("8.2.2(a)")
for n in [10, 100, 1000]:
    z = rng.standard_normal((100_000, n)).max(1) if n<=100 else np.concatenate([rng.standard_normal((10_000, n)).max(1) for _ in range(10)])
    ex = quad(lambda x: x*n*norm.pdf(x)*norm.cdf(x)**(n-1), -10, 10)[0]
    print(f" n={n}: MC {z.mean():.4f} +- {z.std()/np.sqrt(len(z)):.4f}, exact {ex:.4f}, sqrt(2 ln n) {np.sqrt(2*np.log(n)):.4f}, ratio {z.mean()/np.sqrt(2*np.log(n)):.3f}")
for n in [1e4,1e6,1e9]:
    ex = quad(lambda x: x*n*norm.pdf(x)*np.exp((n-1)*norm.logcdf(x)), -10, 12, points=[np.sqrt(2*np.log(n))], limit=200)[0]
    print(f" n={n:.0e}: exact {ex:.4f}, sqrt(2 ln n) {np.sqrt(2*np.log(n)):.4f}, ratio {ex/np.sqrt(2*np.log(n)):.3f}")
print("8.2.2(b)")
p = norm.cdf(0.55/0.5) - norm.cdf(0.45/0.5)
print(f" P(pred in [1.35,1.45] | truth off by 0.5) = {p:.5f}; candidates for 66: {66/p:.0f}")
p0 = norm.cdf(0.05/0.5) - norm.cdf(-0.05/0.5)
print(f" (for comparison truth exactly 1.4: P = {p0:.4f})")
# posterior: of those landing in window, if fraction f of candidates truly at 1.4
for f in [0.01, 0.05]:
    post = f*p0/(f*p0+(1-f)*p)
    print(f" prior frac true-hits {f}: posterior P(true 1.4 | in window) = {post:.3f}")
# grid-size scaling: expected false hits ∝ n
