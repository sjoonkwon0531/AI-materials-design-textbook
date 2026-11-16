"""7.1.3 depth sim (fixed), 7.1.4 long-run Adam+L2, 7.1.5 ensemble, 7.1.6 exact SVD norms."""
import math, numpy as np, torch
from scipy import integrate
torch.set_num_threads(1)
silu = lambda z: z*0.5*(1+np.tanh(z/2)); phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
m2 = integrate.quad(lambda z: silu(z)**2*phi(z), -np.inf, np.inf)[0]; c = 1/m2
def ratio(q): return integrate.quad(lambda z: silu(np.sqrt(q)*z)**2*phi(z), -np.inf, np.inf)[0]/q
print("=== 7.1.3 q-map iteration q_{l+1} = c q ratio(q), c=%.4f" % c)
for q0 in [0.9, 0.99, 1.0, 1.01, 1.1]:
    q = q0; qs = []
    for l in range(30): q = c*q*ratio(q); qs.append(q)
    print(f"  q0={q0}: q at layer 10,20,30 = {qs[9]:.3g}, {qs[19]:.3g}, {qs[29]:.3g}")
print("  asymptotic factors c/2, c/4 =", c/2, c/4)
rng = np.random.default_rng(0); n = 512
for s in range(3):
    rng = np.random.default_rng(s); z = rng.standard_normal((n, 1000)); out = []
    for l in range(30):
        W = rng.normal(0, np.sqrt(c/n), (n, n)); z = W @ silu(z); out.append(z.var())
    print(f"  width 512 sim seed {s}: Var(z) at layer 1,5,10,20,30 =", [f"{out[k-1]:.3g}" for k in (1, 5, 10, 20, 30)])

print("=== 7.1.4 Adam+L2, zero data gradient, long run (float64 and float32)")
for dt_ in [torch.float64, torch.float32]:
    th = torch.tensor([1.0], dtype=dt_, requires_grad=True)
    opt = torch.optim.Adam([th], lr=1e-3, weight_decay=0.5); tr = [1.0]; ratios = []
    for t in range(1, 20001):
        opt.zero_grad(); th.grad = torch.zeros(1, dtype=dt_); opt.step(); tr.append(th.item())
    tr = np.array(tr)
    neg = np.where(tr <= 0)[0]
    print(f"  {dt_}: theta at 1000,1500,2000,3000,5000,10000,20000 =", [f"{tr[k]:.3e}" for k in (1000, 1500, 2000, 3000, 5000, 10000, 20000)],
          "first theta<=0:", (neg[0] if len(neg) else None))
tr64 = None
# step ratio m_hat/sqrt(v_hat) manual reproduction
b1, b2, eta, lam = 0.9, 0.999, 1e-3, 0.5
th, m, v = 1.0, 0.0, 0.0; R = {}
for t in range(1, 4001):
    g = lam*th; m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g*g
    r = (m/(1-b1**t))/(math.sqrt(v/(1-b2**t))+1e-8); th -= eta*r
    if t in (1, 10, 100, 300, 500, 700, 1000, 1500, 2000, 3000, 4000): R[t] = (r, th)
print("  manual: t: ratio m^/sqrt(v^), theta ->", {k: (round(a, 3), f"{b:.3e}") for k, b in [(k, R[k][1]) for k in R] for a in [R[k][0]]})
# local decay rate
th, m, v = 1.0, 0.0, 0.0; hist = []
for t in range(1, 20001):
    g = lam*th; m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g*g
    th -= eta*(m/(1-b1**t))/(math.sqrt(v/(1-b2**t))+1e-8); hist.append(th)
hist = np.array(hist)
for a, b in [(1000, 1500), (1800, 2000), (2500, 3000), (4000, 5000)]:
    print(f"  decay rate -dln(theta)/dt over [{a},{b}] = {math.log(hist[a-1]/hist[b-1])/(b-a):.2e}")
print("  (1-beta2)/2 =", (1-b2)/2, " -ln(beta2)/2 =", -math.log(b2)/2)

print("=== 7.1.5 ensemble over starting points")
F0 = 1.0
def vv(x, v, dt, T):
    E0 = 0.5*v*v+F0*abs(x); f = -F0*np.sign(x); emax = 0.0; errs = []
    for _ in range(int(round(T/dt))):
        vh = v+0.5*dt*f; xn = x+dt*vh; fn = -F0*np.sign(xn); v = vh+0.5*dt*fn
        if np.sign(xn) != np.sign(x):
            errs.append(0.5*v*v+F0*abs(xn) - (0.5*vh0**2 if False else 0))
        x = xn; f = fn; emax = max(emax, abs(0.5*v*v+F0*abs(x)-E0))
    return emax
rng = np.random.default_rng(0); x0s = rng.uniform(-1.5, -0.5, 40)
dts = [0.1, 0.05, 0.025, 0.0125]; med = []
for dt in dts:
    e = [vv(x0, 0.0, dt, 50.0) for x0 in x0s]; med.append(np.median(e))
    print(f"  dt={dt}: median over 40 starts of max|E-E0| (T=50) = {np.median(e):.3e}; mean = {np.mean(e):.3e}")
print("  slope (median) =", np.polyfit(np.log(dts), np.log(med), 1)[0])
# per-crossing statistics: single crossing error vs s
dt = 0.05; v0 = 1.0
for s in [0.1, 0.25, 0.5, 0.75, 0.9]:
    x0 = -s*v0*dt; vh = v0+0.5*dt*F0; x1 = x0+dt*vh; v1 = vh-0.5*dt*F0
    dE = 0.5*v1**2+F0*abs(x1)-0.5*v0**2-F0*abs(x0)
    print(f"  s={s}: dE={dE:+.6f}  F0 v0 dt (1-2s) + F0^2 dt^2/2 = {F0*v0*dt*(1-2*s)+F0**2*dt**2/2:+.6f}")

print("=== 7.1.6 exact SVD of Phi near N=n (3 first-layer seeds)")
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
df = pd.read_parquet("../ch07/data/magpie_expt_gap.parquet")
X = df.drop(columns=["formula", "target"]).values; y = df.target.values
trn, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
rng = np.random.default_rng(0); n = 600; sub = rng.choice(trn, n, replace=False)
vt = VarianceThreshold(1e-6).fit(X[sub]); sc = StandardScaler().fit(vt.transform(X[sub]))
Xs = np.clip(sc.transform(vt.transform(X[sub])), -5, 5); ys = y[sub]-y[sub].mean(); p = Xs.shape[1]
for N in [100, 300, 550, 590, 600, 610, 650, 1000, 5000, 30000]:
    smin, smax, wex, wlab, tail = [], [], [], [], []
    for s in range(3):
        W = np.random.default_rng(100+s).normal(0, 1/np.sqrt(p), (p, N)); Phi = np.maximum(Xs @ W, 0)
        U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
        proj = (U.T @ ys)**2; contrib = proj/S**2
        smin.append(S[-1]); smax.append(S[0]); wex.append(np.sqrt(contrib.sum())); tail.append(contrib[-1]/contrib.sum())
        K = Phi @ Phi.T; al = np.linalg.pinv(K, rcond=1e-12, hermitian=True) @ ys; wlab.append(np.sqrt(max(al @ K @ al, 0)))
    print(f"  N={N:6d}: median sigma_min={np.median(smin):.2e} sigma_max={np.median(smax):.1f}  ||w|| exact={np.median(wex):.3g}"
          f" (Lab pinv(K,1e-12): {np.median(wlab):.3g}); share of smallest-sigma term {np.median(tail):.2f}")
