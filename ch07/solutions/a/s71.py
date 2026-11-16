"""Solutions 7.1.x numerics."""
import math, numpy as np, torch
from scipy import integrate
torch.set_num_threads(1)
print("=== 7.1.2 parameter / MAC counts")
L = [132, 256, 128, 1]
P = sum(a*b + b for a, b in zip(L[:-1], L[1:])); fw = sum(a*b for a, b in zip(L[:-1], L[1:]))
bw = fw + sum(a*b for a, b in zip(L[1:-1], L[2:]))
print("params", P, "fwd MAC/sample", fw, "bwd MAC", bw, "total", fw+bw, "ratio", (fw+bw)/fw,
      "batch64 MAC", 64*(fw+bw), "GFLOP", 2*64*(fw+bw)/1e9)

print("=== 7.1.3")
silu = lambda z: z/(1+np.exp(-z)); phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
m2 = integrate.quad(lambda z: silu(z)**2*phi(z), -np.inf, np.inf)[0]
m1 = integrate.quad(lambda z: silu(z)*phi(z), -np.inf, np.inf)[0]
print(f"E[SiLU^2]={m2:.4f}  E[SiLU]={m1:.4f}  Var={m2-m1**2:.4f}  1/E[SiLU^2]={1/m2:.3f}")
for q in [0.01, 0.1, 0.5, 1, 2, 5, 25]:
    r = integrate.quad(lambda z: silu(np.sqrt(q)*z)**2*phi(z), -np.inf, np.inf)[0] / q
    print(f"  q={q:5.2f}: E[SiLU(sqrt(q) z)^2]/q = {r:.4f}")
c = 1/m2
# deep net simulation: width 256, depth 30, Var(W)=c/n, zero bias
rng = np.random.default_rng(0); n = 256
x = rng.standard_normal((n, 2000)); z = x.copy(); out = []
for l in range(30):
    W = rng.normal(0, np.sqrt(c/n), (n, n)); z = W @ silu(z) if l > 0 else W @ x
    out.append(z.var())
print("  depth sim (width 256, c=%.3f): Var(z_l) at l=1,2,5,10,20,30:" % c, [round(out[k-1], 3) for k in (1, 2, 5, 10, 20, 30)])
# fixed point of q -> c E[silu(sqrt q z)^2]
def g(q): return c*integrate.quad(lambda z: silu(np.sqrt(q)*z)**2*phi(z), -np.inf, np.inf)[0]
h = 1e-4; print("  map slope dg/dq at q=1:", (g(1+h)-g(1-h))/(2*h))
# input correlation: mean of silu gives correlated pre-activations
print("  E[a]^2/E[a^2] =", m1**2/m2)

print("=== 7.1.4 Adam+L2 with zero data gradient")
def run(kind, lam, steps):
    th = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([th], lr=1e-3, weight_decay=lam) if kind == "L2" else torch.optim.AdamW([th], lr=1e-3, weight_decay=lam)
    tr = [1.0]
    for _ in range(steps):
        opt.zero_grad(); th.grad = torch.zeros(1, dtype=torch.float64); opt.step(); tr.append(th.item())
    return np.array(tr)
for lam in [0.5, 0.05, 5.0]:
    tr = run("L2", lam, 2000)
    first0 = int(np.argmax(tr <= 0))
    print(f"  lam={lam}: step1 change={tr[1]-tr[0]:.10f}; theta@500={tr[500]:.4f} @900={tr[900]:.4f}; first theta<=0 at step {first0};"
          f" |theta|@2000={abs(tr[2000]):.4f}; max|theta| over 1500-2000 = {np.abs(tr[1500:]).max():.4f}")
    if lam == 0.5:
        st = np.diff(tr)
        print("   step size at t=1,100,500,900,990:", [f"{st[k]:.2e}" for k in (0, 99, 499, 899, 989)])
        print("   theta at 1000..1100 every 20:", [f"{tr[k]:+.4f}" for k in range(1000, 1101, 20)])
        print("   theta at 1800..2000 every 25:", [f"{tr[k]:+.4f}" for k in range(1800, 2001, 25)])
        # sign changes after first crossing
        s = np.sign(tr[first0:]); print("   sign changes after crossing:", int((np.diff(s) != 0).sum()))
for lam in [0.5, 0.05]:
    t = -1/math.log(1-1e-3*lam); trw = run("W", lam, int(t)+2)
    print(f"  AdamW lam={lam}: exact steps to e^-1 = {t:.1f}; simulated theta at {int(round(t))} = {trw[int(round(t))]:.5f}")
print("  (1-5e-4)^2000 =", (1-5e-4)**2000)

print("=== 7.1.5 velocity Verlet in V=F0|x|")
F0 = 1.0
def vv(x, v, dt, T):
    F = lambda x: -F0*np.sign(x); E = lambda x, v: 0.5*v*v + F0*abs(x)
    E0 = E(x, v); f = F(x); emax = 0.0; ncross = 0; steps = int(round(T/dt))
    for _ in range(steps):
        vh = v + 0.5*dt*f; xn = x + dt*vh; fn = F(xn); v = vh + 0.5*dt*fn
        ncross += (np.sign(xn) != np.sign(x)); x = xn; f = fn
        emax = max(emax, abs(E(x, v)-E0))
    return emax, ncross
# exact single-step formula check
x0, v0, dt = -0.03, 1.0, 0.1
vh = v0 + 0.5*dt*F0; x1 = x0 + dt*vh; v1 = vh - 0.5*dt*F0
dE = (0.5*v1**2 + F0*abs(x1)) - (0.5*v0**2 + F0*abs(x0))
print(f"  single step x0=-0.03,v0=1,dt=0.1: dE={dE:.6f}  formula F0(2x0+v0 dt+F0 dt^2/2)={F0*(2*x0+v0*dt+F0*dt**2/2):.6f}")
res = []
for dt in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
    e, nc = vv(-1.0, 0.0, dt, 100.0); res.append(e)
    print(f"  dt={dt:.5f}: max|E-E0| over T=100 = {e:.3e}, crossings {nc}")
sl = np.polyfit(np.log([0.1, 0.05, 0.025, 0.0125, 0.00625]), np.log(res), 1)[0]
print(f"  log-log slope = {sl:.2f}")
# mean of g(s) over uniform phase
print("  mean of g(s)=1-2s over s in (0,1):", integrate.quad(lambda s: 1-2*s, 0, 1)[0])

print("=== 7.1.6 singular values of the random-feature matrix (Lab 7.4 setting)")
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
df = pd.read_parquet("../../../ch07/data/magpie_expt_gap.parquet")
X = df.drop(columns=["formula", "target"]).values; y = df.target.values
tr, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))
rng = np.random.default_rng(0); n = 600; sub = rng.choice(tr, n, replace=False)
vt = VarianceThreshold(1e-6).fit(X[sub]); sc = StandardScaler().fit(vt.transform(X[sub]))
Xs = np.clip(sc.transform(vt.transform(X[sub])), -5, 5); ys = y[sub] - y[sub].mean(); p = Xs.shape[1]
for N in [300, 550, 590, 600, 610, 650, 1000, 5000, 30000]:
    W = np.random.default_rng(100).normal(0, 1/np.sqrt(p), (p, N)); Phi = np.maximum(Xs @ W, 0)
    K = Phi @ Phi.T; ev, U = np.linalg.eigh(K); ev = np.clip(ev, 0, None)
    r = min(N, n); evs = ev[::-1][:r]; Us = U[:, ::-1][:, :r]
    proj = (Us.T @ ys)**2
    if N < n:
        wn = None
    contrib = proj/np.maximum(evs, 1e-300)
    print(f"  N={N:6d}: sigma_min={np.sqrt(evs[-1]):.3e}  sigma_max={np.sqrt(evs[0]):.3e}  ||w||^2 from formula={contrib.sum():.3e}"
          f"  share of 5 smallest sigma terms={contrib[-5:].sum()/contrib.sum():.3f}")
# GD from zero converges to min-norm (small toy)
rng = np.random.default_rng(1); A = rng.standard_normal((20, 50)); b = rng.standard_normal(20)
w = np.zeros(50); eta = 1/np.linalg.norm(A, 2)**2
for _ in range(20000): w -= eta*A.T@(A@w-b)
wmn = np.linalg.pinv(A)@b
w2 = rng.standard_normal(50); w2s = w2.copy()
for _ in range(20000): w2 -= eta*A.T@(A@w2-b)
print(f"  GD from 0: |w-w_mn|={np.linalg.norm(w-wmn):.1e}, |w|={np.linalg.norm(w):.4f}, |w_mn|={np.linalg.norm(wmn):.4f};"
      f" GD from random w0: |w-w_mn|={np.linalg.norm(w2-wmn):.3f}, equals |P_null w0|={np.linalg.norm(w2s - np.linalg.pinv(A)@(A@w2s)):.3f}")
