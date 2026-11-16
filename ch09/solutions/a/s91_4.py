"""Ex 9.1.4: separable HNN H = T(p) + V(q); velocity-Verlet vs RK4 at h = 0.2."""
import importlib.util, sys, time, math
import numpy as np, torch
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("hnn", "../ch09/labs/91_hnn.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
torch.set_default_dtype(torch.float64)
H_true = L.H_true

def mlp1(w=64):
    return torch.nn.Sequential(torch.nn.Linear(1, w), torch.nn.Tanh(), torch.nn.Linear(w, w), torch.nn.Tanh(), torch.nn.Linear(w, 1))

class SepHNN(torch.nn.Module):
    def __init__(s, w=64):
        super().__init__(); s.T = mlp1(w); s.V = mlp1(w)
    def H(s, x):
        return (s.T(x[:, 1:2]) + s.V(x[:, 0:1])).squeeze(-1)
    def forward(s, x):
        with torch.enable_grad():
            if not x.requires_grad: x = x.clone().requires_grad_(True)
            dH = torch.autograd.grad(s.H(x).sum(), x, create_graph=True)[0]
        return torch.stack([dH[:, 1], -dH[:, 0]], 1)
    def dV(s, q):
        q = torch.tensor([[q]], requires_grad=True); return float(torch.autograd.grad(s.V(q).sum(), q)[0])
    def dT(s, p):
        p = torch.tensor([[p]], requires_grad=True); return float(torch.autograd.grad(s.T(p).sum(), p)[0])

def vv(dV, dT, x0, h, n):
    q, p = x0; out = [(q, p)]
    for _ in range(n):
        p = p - 0.5 * h * dV(q); q = q + h * dT(p); p = p - 0.5 * h * dV(q); out.append((q, p))
    return np.array(out)

def rk4_np(f, x0, h, n):
    x = np.array(x0, float); out = [x.copy()]
    for _ in range(n):
        k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(x.copy())
    return np.array(out)

x0 = [2.0, 0.0]; T = 200.0; E0 = H_true(*x0)
ftrue = lambda y: np.array([y[1], -np.sin(y[0])])
print(f"E0 = {E0:.4f}")
# integrator-only shadows on the TRUE field
for h in (0.2, 0.02):
    n = int(round(T / h))
    tv = vv(np.sin, lambda p: p, x0, h, n); E = H_true(tv[:, 0], tv[:, 1])
    tr = rk4_np(ftrue, x0, h, n); Er = H_true(tr[:, 0], tr[:, 1])
    print(f"true field h={h}: VV max|E-E0| {np.abs(E-E0).max():.3e} (E(T)-E0 {E[-1]-E0:+.2e}, range [{E.min()-E0:+.4f},{E.max()-E0:+.4f}]) | RK4 max|E-E0| {np.abs(Er-E0).max():.3e} (E(T)-E0 {Er[-1]-E0:+.3e})")
for seed in [int(a) for a in sys.argv[1:]] or [0]:
    torch.manual_seed(seed); m = SepHNN(); c0 = time.process_time(); loss = L.fit(m, seed); cpu = time.process_time() - c0
    qq, pp = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-1.5, 1.5, 31)); msk = H_true(qq, pp) < 1.8
    xg = torch.tensor(np.c_[qq[msk], pp[msk]]); fg = m(xg).detach().numpy(); ft = np.c_[pp[msk], -np.sin(qq[msk])]
    print(f"sepHNN seed {seed}: train loss {loss:.2e}, field RMSE {np.sqrt(((fg-ft)**2).sum(1).mean()):.4f}, train cpu {cpu:.0f} s, params {sum(p.numel() for p in m.parameters())}")
    fm = lambda y: m(torch.tensor(y[None])).detach().numpy()[0]
    Hth = lambda tr: m.H(torch.tensor(tr)).detach().numpy()
    res = {}
    for name, tr in (("VV   h=0.2 ", vv(m.dV, m.dT, x0, 0.2, 1000)), ("RK4  h=0.2 ", rk4_np(fm, x0, 0.2, 1000)), ("RK4  h=0.02", rk4_np(fm, x0, 0.02, 10000)), ("VV   h=0.02", vv(m.dV, m.dT, x0, 0.02, 10000))):
        E = H_true(tr[:, 0], tr[:, 1]); Hh = Hth(tr)
        n4 = len(E) // 4
        print(f"  {name}: max|E-E0| {np.abs(E-E0).max():.4f}, E range [{E.min()-E0:+.4f},{E.max()-E0:+.4f}], E(T)-E0 {E[-1]-E0:+.4f}, mean E-E0 first/last quarter {E[:n4].mean()-E0:+.4f}/{E[-n4:].mean()-E0:+.4f}; max|H_th-H_th(0)| {np.abs(Hh-Hh[0]).max():.2e}, H_th(T)-H_th(0) {Hh[-1]-Hh[0]:+.2e}")
    # learned potentials vs truth (up to constants)
    qs = np.linspace(-2, 2, 81); ps = np.linspace(-1.5, 1.5, 61)
    dVg = np.array([m.dV(q) for q in qs]); dTg = np.array([m.dT(p) for p in ps])
    print(f"  max|V'_th - sin q| on [-2,2]: {np.abs(dVg-np.sin(qs)).max():.4f}; max|T'_th - p| on [-1.5,1.5]: {np.abs(dTg-ps).max():.4f}")

    # long run: bounded (symplectic) vs drifting (RK4) at h = 0.2
    TL = 4000.0; nL = int(TL / 0.2)
    for name, tr in (("VV   h=0.2 ", vv(m.dV, m.dT, x0, 0.2, nL)), ("RK4  h=0.2 ", rk4_np(fm, x0, 0.2, nL))):
        E = H_true(tr[:, 0], tr[:, 1]); Hh = Hth(tr)
        marks = [1000, 5000, 10000, 20000]
        print(f"  long {name}: " + "; ".join(f"T={k*0.2:.0f}: max|E-E0| {np.abs(E[:k+1]-E0).max():.4f}, H_th-H_th(0) {Hh[k]-Hh[0]:+.4f}" for k in marks), flush=True)
    torch.save(m.state_dict(), f"../ch09/solutions/a/s91_4_sep_{seed}.pt")
