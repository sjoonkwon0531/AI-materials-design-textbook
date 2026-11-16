"""Ex 9.2.2, 9.2.4, 9.2.5: small calculations."""
import numpy as np, math
PI = np.pi
print("== 9.2.2 ==")
for name, eps in (("float64", 2.2e-16), ("float32", 1.2e-7)):
    u, u4 = 1.0, 1e4
    hs = (48 * eps * u / u4) ** 0.25; Es = 2 / math.sqrt(3) * math.sqrt(eps * u * u4)
    E = lambda h: u4 * h * h / 12 + 4 * eps * u / h**2
    print(f"{name}: h* = {hs:.3e}, E* = {Es:.3e} (check {E(hs):.3e}), rel = {Es/39.3:.2e}; model rel at h=1e-2,3e-3,1e-3,1e-4,3e-5: " + ", ".join(f"{E(h)/39.3:.1e}" for h in (1e-2, 3e-3, 1e-3, 1e-4, 3e-5)))
# (c) fourth derivative by central differences in float32 / float64, u = sin(2 pi x)
x = np.linspace(0.05, 0.95, 19)
for dt in (np.float64, np.float32):
    best = (1e9, 0); rows = []
    for h in (1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 1e-3):
        xx = x.astype(dt); hh = dt(h); f = lambda z: np.sin(dt(2 * PI) * z)
        d4 = (f(xx + 2 * hh) - 4 * f(xx + hh) + 6 * f(xx) - 4 * f(xx - hh) + f(xx - 2 * hh)) / hh**4
        ex = (2 * PI) ** 4 * np.sin(2 * PI * x)
        e = np.abs(d4.astype(float) - ex).max() / np.abs(ex).max(); rows.append(f"h={h:g}: {e:.1e}")
        if e < best[0]: best = (e, h)
    print(dt.__name__, "u''''(sin 2 pi x) max rel err:", "; ".join(rows), "| best", best)
eps = 1.2e-7; u6 = (2 * PI) ** 6; h6 = (192 * eps / u6) ** (1 / 6); print(f"model 4th-derivative: h* = {h6:.3e}, E* = {0.25*u6*h6**2:.2f}, rel to (2pi)^4: {0.25*u6*h6**2/(2*PI)**4:.2e}; rounding factor from h*/3: {3**4}")
print("== 9.2.4 ==")
for s in (1, 10, 50):
    print(f"sigma={s}: 3(2 pi s)^4 = {3*(2*PI*s)**4:.3e}; 105 (2 pi s)^8 = {105*(2*PI*s)**8:.3e}")
m = 4096; print(f"relative std of sample mean of b^4 with m={m}: {math.sqrt(105-9)/3/math.sqrt(m):.3f}; script/theory = {4.518e3/4.676e3:.4f}")
# (d) NTK weights on the linear RFF model: traces equalised, conditioning inside K_rr unchanged
rng = np.random.default_rng(0); b = 10 * rng.standard_normal(64); xr = np.linspace(0, 1, 258)[1:-1]; xb = np.array([0.0, 1.0])
feat = lambda x: np.concatenate([np.cos(2 * PI * np.outer(x, b)), np.sin(2 * PI * np.outer(x, b))], 1)
Ju = feat(xb); Jr = feat(xr) * np.tile((2 * PI * b) ** 2, 2)
Kuu, Krr = Ju @ Ju.T, Jr @ Jr.T; trK = np.trace(Kuu) + np.trace(Krr); lb, lr = trK / np.trace(Kuu), trK / np.trace(Krr)
ev = np.linalg.eigvalsh(Krr); evp = ev[ev > ev.max() * 1e-14]
print(f"RFF m=64 pairs sigma=10: Tr Kuu {np.trace(Kuu):.3e}, Tr Krr {np.trace(Krr):.3e}; lam_b {lb:.3e}, lam_r {lr:.6f}, lam_b/lam_r {lb/lr:.3e}")
print(f"  after weighting: Tr(lam_b Kuu) {lb*np.trace(Kuu):.6e} = Tr(lam_r Krr) {lr*np.trace(Krr):.6e}")
print(f"  K_rr eigenvalues: max {ev.max():.3e}, smallest nonzero {evp.min():.3e} ({len(evp)} nonzero), ratio {ev.max()/evp.min():.3e}; after weighting max {lr*ev.max():.3e} min {lr*evp.min():.3e}, ratio unchanged")
print(f"  (2 pi b)^4 range over the 64 features: {((2*PI*b)**4).min():.3e} .. {((2*PI*b)**4).max():.3e}; 25^4 = {25**4}")
print("== 9.2.5 ==")
d = 0.02; Dt = 0.5; sD = math.sqrt(2) * d / (PI**2 * Dt); print(f"sigma_D = {sD:.5f} ({100*sD/0.1:.1f} % of D=0.1); B2 PINN 0.0029 -> ratio {sD/0.0029:.2f}; sine-series LS 0.0012 -> ratio {sD/0.0012:.2f}; sqrt(40) = {math.sqrt(40):.2f}")
# Monte Carlo check of the two-point formula
rng = np.random.default_rng(1); D = 0.1; t1, t2 = 0.25, 0.75
y1 = np.exp(-D*PI**2*t1) * (1 + d * rng.standard_normal(200000)); y2 = np.exp(-D*PI**2*t2) * (1 + d * rng.standard_normal(200000))
Dh = np.log(y1 / y2) / (PI**2 * (t2 - t1)); print(f"MC two-point estimator: mean {Dh.mean():.5f}, std {Dh.std():.5f}")
# Cramer-Rao bounds for the Lab B2 design: sensors x = .15,.3,.5,.7,.85; t = 16 times in [0.1, 1]; absolute noise 0.02
xs = np.array([0.15, 0.3, 0.5, 0.7, 0.85]); ts = np.linspace(0.1, 1.0, 16); X, Tt = np.meshgrid(xs, ts, indexing="ij"); X, Tt = X.ravel(), Tt.ravel(); sig = 0.02
def crb(modes, amps):
    cols = [np.sin(k*PI*X) * np.exp(-D*(k*PI)**2*Tt) for k in modes]                       # d/dA_k
    dD = sum(a * (-(k*PI)**2 * Tt) * np.sin(k*PI*X) * np.exp(-D*(k*PI)**2*Tt) for k, a in zip(modes, amps))
    J = np.c_[np.array(cols).T, dD] / sig; C = np.linalg.inv(J.T @ J); return math.sqrt(C[-1, -1])
print(f"CRB sigma_D, 80 points: one mode (A1, D unknown) {crb([1],[1.0]):.5f}; two modes (A1, A3, D) {crb([1,3],[1.0,0.5]):.5f}; four modes (A1..A4, D) {crb([1,2,3,4],[1.0,0.0,0.5,0.0]):.5f}")
u = lambda x, t: np.exp(-D*PI**2*t)*np.sin(PI*x) + 0.5*np.exp(-9*D*PI**2*t)*np.sin(3*PI*x)
print(f"signal at sensors: t=0.1: {[round(float(u(x,0.1)),3) for x in xs]}, t=1: {[round(float(u(x,1.0)),3) for x in xs]}; relative noise 0.02/u at t=1: {[round(0.02/float(u(x,1.0)),3) for x in xs]}")
print(f"(d) sin(3 pi/3) = {math.sin(PI):.1e}, sin(3*2pi/3) = {math.sin(2*PI):.1e}; sin(pi/3) = {math.sin(PI/3):.4f}; modes invisible: k = 3, 6, 9, ...")
