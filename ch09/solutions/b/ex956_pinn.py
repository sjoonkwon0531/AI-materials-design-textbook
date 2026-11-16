"""Exercise 9.5.6: variants of the hard-initial-condition PINN of Lab 9.18 for one Navier-Stokes instance (test trajectory 0).
  feat  : single window t in [0, 5], extra input features sin 2pi(x+y), cos 2pi(x+y)        (part c)
  march : five windows of length 1 solved in sequence (time marching), warm start              (part b)
Usage: python ex956_pinn.py feat|march [n_adam per window] [n_lbfgs per window] [extra: 0|1] [resample: 0|1]"""
import sys, time, math, importlib.util
import numpy as np, torch
LABS = "../ch09/labs/"
spec = importlib.util.spec_from_file_location("ns", LABS + "95_ns_solver.py")
ns = importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)
torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
MODE = sys.argv[1]
N_ADAM = int(sys.argv[2]) if len(sys.argv) > 2 else (2000 if MODE == "feat" else 400)
N_LBFGS = int(sys.argv[3]) if len(sys.argv) > 3 else (300 if MODE == "feat" else 60)
EXTRA = bool(int(sys.argv[4])) if len(sys.argv) > 4 else (MODE == "feat")
RESAMPLE = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
NU, K, NF, KC = 1e-3, 5, 2048, 12
traj = np.load(LABS + "../data/95_ns_nu1e-3_64.npz")["w"][200].astype(np.float64)       # test trajectory 0
xg = torch.arange(64, dtype=torch.float32) / 64
X0, Y0 = torch.meshgrid(xg, xg, indexing="ij"); x0, y0 = X0.reshape(-1, 1), Y0.reshape(-1, 1)
kk = 2 * math.pi * torch.arange(1, K + 1, dtype=torch.float32).view(1, -1)
tp = 2 * math.pi
kf = np.fft.fftfreq(64, d=1 / 64); KXf, KYf = np.meshgrid(kf, kf, indexing="ij")
keep = (np.abs(KXf) <= KC) & (np.abs(KYf) <= KC)
kxs = torch.tensor(tp * KXf[keep], dtype=torch.float32); kys = torch.tensor(tp * KYf[keep], dtype=torch.float32)
grad = lambda a, b: torch.autograd.grad(a.sum(), b, create_graph=True)[0]
forcing = lambda x, y: 0.1 * (torch.sin(tp * (x + y)) + torch.cos(tp * (x + y)))


class Net(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        dims = [4 * K + 1 + (2 if EXTRA else 0)] + [width] * depth + [1]
        mlp = lambda: torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.psi_net, self.w_net = mlp(), mlp()
    def forward(self, x, y, tau):
        f = [torch.sin(kk * x), torch.cos(kk * x), torch.sin(kk * y), torch.cos(kk * y)]
        if EXTRA: f += [torch.sin(tp * (x + y)), torch.cos(tp * (x + y))]
        feat = torch.cat(f + [tau], 1); out = []
        for layers in (self.psi_net, self.w_net):
            h = feat
            for l in layers[:-1]: h = torch.tanh(l(h))
            out.append(layers[-1](h))
        return out


def solve_window(net, w_start, Tw, n_adam, n_lbfgs):
    """Hard initial condition w = w_start + (t/Tw) a(x, y, t/Tw), t in [0, Tw].  Returns (predict(t), log)."""
    coef = torch.tensor((np.fft.fft2(w_start) / 64**2)[keep], dtype=torch.complex64)
    def series(x, y):
        E = torch.exp(1j * (x * kxs.view(1, -1) + y * kys.view(1, -1))); ev = lambda c: (E @ c).real.unsqueeze(1)
        return [v.detach() for v in (ev(coef), ev(1j * kxs * coef), ev(1j * kys * coef), ev(-(kxs**2 + kys**2) * coef))]
    def residuals(pts, w0p):
        x, y, t = [p.clone().requires_grad_(True) for p in pts]
        tau = t / Tw; psi, a = net(x, y, tau)
        a_x, a_y, a_t = grad(a, x), grad(a, y), grad(a, t)
        w = w0p[0] + tau * a
        w_x, w_y, w_t = w0p[1] + tau * a_x, w0p[2] + tau * a_y, a / Tw + tau * a_t
        lap_w = w0p[3] + tau * (grad(a_x, x) + grad(a_y, y))
        psi_x, psi_y = grad(psi, x), grad(psi, y)
        return w_t + psi_y * w_x - psi_x * w_y - NU * lap_w - forcing(x, y), grad(psi_x, x) + grad(psi_y, y) + w
    sample = lambda: (torch.rand(NF, 1), torch.rand(NF, 1), Tw * torch.rand(NF, 1))
    def loss_fn(pts, w0p):
        r1, r2 = residuals(pts, w0p); return (r1**2).mean() + (r2**2).mean(), (r1**2).mean().item(), (r2**2).mean().item()
    pts = sample(); w0p = series(pts[0], pts[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_adam, eta_min=1e-4)
    for it in range(n_adam):
        if RESAMPLE: pts = sample(); w0p = series(pts[0], pts[1])
        opt.zero_grad(); L, l1, l2 = loss_fn(pts, w0p); L.backward(); opt.step(); sched.step()
    if n_lbfgs > 0:
        lb = torch.optim.LBFGS(net.parameters(), max_iter=n_lbfgs, history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        def closure():
            lb.zero_grad(); L = loss_fn(pts, w0p)[0]; L.backward(); return L
        lb.step(closure)
    _, l1, l2 = loss_fn(pts, w0p)
    torch.manual_seed(123); fresh = sample(); lf = loss_fn(fresh, series(fresh[0], fresh[1]))[1]
    w0g = series(x0, y0)[0]
    def predict(t):
        with torch.no_grad():
            return (w0g + (t / Tw) * net(x0, y0, torch.full_like(x0, t / Tw))[1]).numpy().reshape(64, 64).astype(float)
    return predict, (l1, l2, lf)

net = Net(); c0 = time.process_time()
print(f"mode {MODE}: {sum(p.numel() for p in net.parameters()):,d} parameters, extra features {EXTRA}, resampling {RESAMPLE}, "
      f"{'1 window of length 5' if MODE == 'feat' else '5 windows of length 1'}, Adam {N_ADAM} + L-BFGS {N_LBFGS} per window", flush=True)
if MODE == "feat":
    predict, (l1, l2, lf) = solve_window(net, traj[0], 5.0, N_ADAM, N_LBFGS)
    preds = [predict(float(t)) for t in range(1, 6)]
    print(f"  loss PDE {l1:.2e} (fresh points {lf:.2e}), Poisson {l2:.2e}")
else:
    w, preds = traj[0], []
    for i in range(5):
        predict, (l1, l2, lf) = solve_window(net, w, 1.0, N_ADAM, N_LBFGS)
        w = predict(1.0); preds.append(w)
        print(f"  window [{i}, {i + 1}]: CPU {time.process_time() - c0:4.0f} s | loss PDE {l1:.2e} (fresh points {lf:.2e}), Poisson {l2:.2e} | rel. L2 at t = {i + 1}: {ns.rel_l2(w, traj[i + 1]):.4f}"
              f" | persistence over this window (true): {ns.rel_l2(traj[i], traj[i + 1]):.3f}", flush=True)
err = [ns.rel_l2(p, traj[i + 1]) for i, p in enumerate(preds)]
print("rel. L2 error vs the 64^2 solver at t = 1..5: " + " ".join(f"{v:.4f}" for v in err))
print("'nothing happens' (w = w0) at t = 1..5:        " + " ".join(f"{ns.rel_l2(traj[0], traj[t]):.4f}" for t in range(1, 6)))
print(f"CPU time {time.process_time() - c0:.0f} s")
