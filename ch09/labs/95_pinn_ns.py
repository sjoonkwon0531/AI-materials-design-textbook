"""Lab 9.5 (c), PINN part: a physics-informed network for ONE Navier-Stokes instance.
Unknowns psi(x, y, t), w(x, y, t) on the torus, mixed (stream function - vorticity) form:
    r1 = w_t + psi_y w_x - psi_x w_y - nu (w_xx + w_yy) - f = 0,     r2 = psi_xx + psi_yy + w = 0,
initial condition w(x, y, 0) = w0 (test trajectory 0 of the Lab 9.5 (b) data), horizon t in [0, T].
Periodicity is built in by feeding sin/cos(2 pi k x), sin/cos(2 pi k y), k = 1..5 (hard constraint).
Two ways to impose the initial condition:
    soft : loss term  LAM_IC * mean (w(x, y, 0) - w0)^2  on the 64^2 grid, collocation points resampled
    hard : ansatz  w = w0(x, y) + (t / T) * a(x, y, t), with w0 given by its Fourier series (|k| <= 12),
           so that the initial condition holds exactly and only the two PDE residuals are minimised
Usage: python 95_pinn_ns.py [T] [adam_iters] [lbfgs_iters] [soft | hard] [output tag]"""
import sys, time, math, importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ns", "95_ns_solver.py")
ns = importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)
torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)

T = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
N_ADAM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
N_LBFGS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MODE = sys.argv[4] if len(sys.argv) > 4 else "soft"
TAG = sys.argv[5] if len(sys.argv) > 5 else ""          # suffix for the output files, e.g. "_long"
NU, K, NF, LAM_IC = 1e-3, 5, 2048, 100.0

d = np.load("../data/95_ns_nu1e-3_64.npz")
traj = d["w"][200].astype(np.float64)                     # test trajectory 0: (21, 64, 64)
w0 = torch.tensor(traj[0], dtype=torch.float32).reshape(-1, 1)
xg = torch.arange(64, dtype=torch.float32) / 64
X0, Y0 = torch.meshgrid(xg, xg, indexing="ij")
x0, y0 = X0.reshape(-1, 1), Y0.reshape(-1, 1)
kk = 2 * math.pi * torch.arange(1, K + 1, dtype=torch.float32).view(1, -1)

class Net(torch.nn.Module):
    """Two separate MLPs, one for psi and one for w (soft) or a (hard). With one shared MLP the
    vorticity-equation residual did not decrease at all in 2000 Adam steps (first attempt)."""
    def __init__(self, width=64, depth=4):
        super().__init__()
        dims = [4 * K + 1] + [width] * depth + [1]
        mlp = lambda: torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.psi_net, self.w_net = mlp(), mlp()
    def forward(self, x, y, t):
        feat = torch.cat([torch.sin(kk * x), torch.cos(kk * x), torch.sin(kk * y), torch.cos(kk * y), t / T], 1)
        out = []
        for layers in (self.psi_net, self.w_net):
            h = feat
            for l in layers[:-1]:
                h = torch.tanh(l(h))
            out.append(layers[-1](h))
        return out[0], out[1]                             # psi, and w (soft) or a (hard)

net = Net()
grad = lambda a, b: torch.autograd.grad(a.sum(), b, create_graph=True)[0]
forcing = lambda x, y: 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))

# ------------------------------------------------------------------ hard initial condition: w0 as a Fourier series
KC = 12
kf = np.fft.fftfreq(64, d=1 / 64)
KXf, KYf = np.meshgrid(kf, kf, indexing="ij")
keep = (np.abs(KXf) <= KC) & (np.abs(KYf) <= KC)
coef = torch.tensor((np.fft.fft2(traj[0]) / 64**2)[keep], dtype=torch.complex64)
kxs = torch.tensor(2 * math.pi * KXf[keep], dtype=torch.float32)
kys = torch.tensor(2 * math.pi * KYf[keep], dtype=torch.float32)

def w0_series(x, y):
    """w0 and its derivatives at arbitrary points, from the truncated Fourier series (exact, no autograd)."""
    E = torch.exp(1j * (x * kxs.view(1, -1) + y * kys.view(1, -1)))
    ev = lambda c: (E @ c).real.unsqueeze(1)
    return ev(coef), ev(1j * kxs * coef), ev(1j * kys * coef), ev(-(kxs**2 + kys**2) * coef)

def residuals_soft(x, y, t):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    psi, w = net(x, y, t)
    w_x, w_y, w_t = grad(w, x), grad(w, y), grad(w, t)
    psi_x, psi_y = grad(psi, x), grad(psi, y)
    r1 = w_t + psi_y * w_x - psi_x * w_y - NU * (grad(w_x, x) + grad(w_y, y)) - forcing(x, y)
    r2 = grad(psi_x, x) + grad(psi_y, y) + w
    return r1, r2

def residuals_hard(x, y, t, w0p):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    psi, a = net(x, y, t)
    tau = t / T
    a_x, a_y, a_t = grad(a, x), grad(a, y), grad(a, t)
    w = w0p[0] + tau * a                                   # ansatz: the initial condition is exact
    w_x, w_y, w_t = w0p[1] + tau * a_x, w0p[2] + tau * a_y, a / T + tau * a_t
    lap_w = w0p[3] + tau * (grad(a_x, x) + grad(a_y, y))
    psi_x, psi_y = grad(psi, x), grad(psi, y)
    r1 = w_t + psi_y * w_x - psi_x * w_y - NU * lap_w - forcing(x, y)
    r2 = grad(psi_x, x) + grad(psi_y, y) + w
    return r1, r2

def sample():
    return torch.rand(NF, 1), torch.rand(NF, 1), T * torch.rand(NF, 1)

if MODE == "hard":
    w0_grid = w0_series(x0, y0)[0]
    print(f"hard IC: Fourier series of w0 with |k| <= {KC} ({int(keep.sum())} modes), "
          f"truncation error {float((w0_grid - w0).norm() / w0.norm()):.1e}")
    PTS = sample()                                        # fixed collocation points
    W0P = [v.detach() for v in w0_series(PTS[0], PTS[1])]

def loss_fn(pts):
    if MODE == "hard":
        r1, r2 = residuals_hard(*pts, W0P)
        l_pde, l_psi = (r1**2).mean(), (r2**2).mean()
        return l_pde + l_psi, (l_pde.item(), l_psi.item(), 0.0)
    r1, r2 = residuals_soft(*pts)
    _, w_ic = net(x0, y0, torch.zeros_like(x0))
    l_pde, l_psi, l_ic = (r1**2).mean(), (r2**2).mean(), ((w_ic - w0) ** 2).mean()
    return l_pde + l_psi + LAM_IC * l_ic, (l_pde.item(), l_psi.item(), l_ic.item())

def predict(tq):
    with torch.no_grad():
        out = net(x0, y0, torch.full_like(x0, float(tq)))[1]
        if MODE == "hard":
            out = w0_grid + (tq / T) * out
    return out.numpy().reshape(64, 64)

def errors():
    return np.array([ns.rel_l2(predict(tq).astype(float), traj[tq]) for tq in range(1, int(T) + 1)])

print(f"PINN for one NS instance ({MODE} initial condition): T = {T:g}, two MLPs {4 * K + 1}-64-64-64-64-1 tanh, "
      f"{sum(p.numel() for p in net.parameters()):,d} parameters, {NF} collocation points"
      f" ({'fixed' if MODE == 'hard' else 'resampled, IC on 64^2'})")
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ADAM, eta_min=1e-4)
t0 = time.perf_counter(); c0 = time.process_time(); log = []
for it in range(1, N_ADAM + 1):
    opt.zero_grad(); L, parts = loss_fn(PTS if MODE == "hard" else sample()); L.backward(); opt.step(); sched.step()
    if it % (N_ADAM // 4) == 0:
        e = errors(); log.append((it, time.process_time() - c0, *parts, e[0], e[-1]))
        print(f"  Adam {it:5d}: CPU {log[-1][1]:5.0f} s | loss PDE {parts[0]:.2e}, Poisson {parts[1]:.2e}, IC {parts[2]:.2e}"
              f" | rel. L2 at t=1: {e[0]:.4f}, at t={T:g}: {e[-1]:.4f}", flush=True)
if N_LBFGS > 0:
    pts = PTS if MODE == "hard" else sample()
    lb = torch.optim.LBFGS(net.parameters(), max_iter=N_LBFGS, history_size=50, tolerance_grad=1e-12,
                           tolerance_change=1e-15, line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad(); L, _ = loss_fn(pts); L.backward(); return L
    lb.step(closure)
    L, parts = loss_fn(pts); e = errors(); log.append((N_ADAM + N_LBFGS, time.process_time() - c0, *parts, e[0], e[-1]))
    print(f"  + L-BFGS  : CPU {log[-1][1]:5.0f} s | loss PDE {parts[0]:.2e}, Poisson {parts[1]:.2e}, IC {parts[2]:.2e}"
          f" | rel. L2 at t=1: {e[0]:.4f}, at t={T:g}: {e[-1]:.4f}")
wall, cpu = time.perf_counter() - t0, time.process_time() - c0
e = errors()
print("rel. L2 error vs the 64^2 solver at t = 1.." + f"{int(T)}: " + " ".join(f"{v:.4f}" for v in e))
print(f"this single instance: wall {wall:.0f} s on a shared machine, CPU time {cpu:.0f} s")
if MODE == "hard":                                        # does the residual generalise away from the collocation points?
    torch.manual_seed(1)
    fresh = sample(); w0f = [v.detach() for v in w0_series(fresh[0], fresh[1])]
    r_tr, r_fr = residuals_hard(*PTS, W0P)[0], residuals_hard(*fresh, w0f)[0]
    print(f"vorticity-equation residual (mean square): {(r_tr**2).mean().item():.2e} on the {NF} training points,"
          f" {(r_fr**2).mean().item():.2e} on {NF} fresh points")
pred = np.stack([predict(tq) for tq in range(int(T) + 1)])
torch.save(net.state_dict(), f"../data/95_pinn_ns_T{T:g}_{MODE}{TAG}.pt")
np.savez(f"../data/95_pinn_ns_T{T:g}_{MODE}{TAG}.npz", pred=pred, err=e, wall=wall, cpu=cpu, log=np.array(log))
