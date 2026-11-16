"""Lab 9.2-convection: a PINN failure mode and three remedies (small-scale version of Krishnapriyan et al. 2021).
u_t + beta u_x = 0, x in [0, 2 pi), t in [0, 1], u(x,0) = sin x, periodic.  Exact: u = sin(x - beta t).
Network 2-50-50-50-50-1 tanh; loss = PDE residual + IC + periodic BC (all soft, weights 1).
"""
import math, time, json, sys
import numpy as np, torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
PI2 = 2 * math.pi
xg = np.linspace(0, PI2, 256, endpoint=False); tg = np.linspace(0, 1, 101)
XG, TG = np.meshgrid(xg, tg, indexing="ij")
xg_t = torch.tensor(XG.reshape(-1, 1)); tg_t = torch.tensor(TG.reshape(-1, 1))


class Net(torch.nn.Module):
    def __init__(s, seed, periodic=False, width=50, t0=0.0, t1=1.0):
        super().__init__()
        torch.manual_seed(seed); s.periodic = periodic; s.t0, s.t1 = t0, t1
        nin = 3 if periodic else 2
        L, d = [], nin
        for _ in range(4):
            L += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
        s.net = torch.nn.Sequential(*L, torch.nn.Linear(d, 1))
    def forward(s, x, t):
        if s.periodic:                                   # exact periodicity in x by construction
            return s.net(torch.cat([torch.cos(x), torch.sin(x), t], 1))
        return s.net(torch.cat([x / math.pi - 1, 2 * (t - s.t0) / (s.t1 - s.t0) - 1], 1))      # inputs scaled to [-1, 1]


def points(seed, n_f=1000, n_i=200, n_b=100, t0=0.0, t1=1.0):
    g = torch.Generator().manual_seed(500 + seed)
    xf = PI2 * torch.rand(n_f, 1, generator=g); tf = t0 + (t1 - t0) * torch.rand(n_f, 1, generator=g)
    xi = PI2 * torch.rand(n_i, 1, generator=g); tb = t0 + (t1 - t0) * torch.rand(n_b, 1, generator=g)
    return xf, tf, xi, tb


def make_loss(model, beta, pts, ic_fn=torch.sin, t0=0.0):
    xf, tf, xi, tb = pts
    def loss_fn(parts=False):
        x = xf.clone().requires_grad_(True); t = tf.clone().requires_grad_(True)
        u = model(x, t)
        ut = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        Lr = (ut + beta * ux).pow(2).mean()
        Li = (model(xi, torch.full_like(xi, t0)) - ic_fn(xi)).pow(2).mean()
        Lb = torch.zeros(()) if model.periodic else (model(torch.zeros_like(tb), tb) - model(torch.full_like(tb, PI2), tb)).pow(2).mean()
        return (Lr, Li, Lb) if parts else Lr + Li + Lb
    return loss_fn


def optimize(model, loss_fn, n_adam, n_lbfgs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_adam):
        opt.zero_grad(); loss_fn().backward(); opt.step()
    if n_lbfgs:
        lb = torch.optim.LBFGS(model.parameters(), max_iter=n_lbfgs, history_size=50, tolerance_grad=1e-12,
                               tolerance_change=1e-15, line_search_fn="strong_wolfe")
        def closure():
            lb.zero_grad(); L = loss_fn(); L.backward(); return L
        lb.step(closure)


def evaluate(model, beta):
    with torch.no_grad():
        up = model(xg_t, tg_t).numpy().reshape(XG.shape)
    ex = np.sin(XG - beta * TG)
    return np.linalg.norm(up - ex) / np.linalg.norm(ex), np.linalg.norm(up - ex, axis=0) / np.linalg.norm(ex, axis=0), up


def lax_wendroff(beta, N, cfl=0.8):
    x = np.linspace(0, PI2, N, endpoint=False); dx = PI2 / N
    nt = int(math.ceil(beta * 1.0 / (cfl * dx))); c = beta * (1.0 / nt) / dx
    u = np.sin(x)
    for _ in range(nt):
        up, um = np.roll(u, -1), np.roll(u, 1)
        u = u - 0.5 * c * (up - um) + 0.5 * c * c * (up - 2 * u + um)
    return np.linalg.norm(u - np.sin(x - beta)) / np.linalg.norm(np.sin(x - beta)), nt


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "AB"
    NA, NL = 2400, 600
    t_all = time.perf_counter(); c_all = time.process_time()
    JS = "../ch09/data/92_convection.json"
    out = json.load(open(JS)) if "A" not in part else {}
    if "A" in part:
        print(f"=== A. vanilla PINN vs beta ({NA} Adam + up to {NL} L-BFGS iterations, 1000 collocation points) ===")
        for beta in (1, 10, 20, 30):
            errs = []
            for seed in range(2 if beta == 30 else 1):
                model = Net(seed); lf = make_loss(model, beta, points(seed)); t0 = time.perf_counter()
                optimize(model, lf, NA, NL); w = time.perf_counter() - t0
                e, et, up = evaluate(model, beta); Lr, Li, Lb = [float(v.detach()) for v in lf(parts=True)]
                errs.append(e)
                if seed == 0:
                    out[f"vanilla_{beta}"] = dict(err_t=et.tolist(), u=up[::4, ::2].tolist(), loss=(Lr, Li, Lb))
                    info = f"seed 0: L_r {Lr:.1e} L_ic {Li:.1e} L_bc {Lb:.1e}, err(t=0.1) {et[10]:.2f}, err(t=1) {et[-1]:.2f}, mean|u(t=1)| {np.abs(up[:, -1]).mean():.3f}, {w:.0f} s"
            out[f"vanilla_{beta}"]["errs"] = errs
            print(f"  beta = {beta:2d}: rel L2 = {', '.join(f'{e:.3f}' for e in errs)} ({len(errs)} seed{'s' if len(errs) > 1 else ''}) | {info}")

    if "B" in part:
        beta = 30
        print(f"=== B. remedies at beta = {beta}: every variant uses 3000 loss evaluations in total, like the vanilla runs ===")
        def report(key, label, errs, et, up, cpu, extra=""):
            out[key] = dict(err_t=et.tolist(), u=up[::4, ::2].tolist(), errs=errs)
            print(f"  {label}: rel L2 = {errs[0]:.4f}, {errs[1]:.4f} | seed 0: err(t=0.1) {et[10]:.3f}, err(t=1) {et[-1]:.3f}{extra}, {cpu:.0f} cpu s per run")
        # (1) curriculum with under-trained stages (Adam only until the last stage): fails
        # (2) curriculum whose every stage is finished with L-BFGS: works
        for key, label, sched, na, nl_stage in (("curriculum_adam", "curriculum 6 stages x 400 Adam, L-BFGS (600) only at the end ", [5, 10, 15, 20, 25, 30], 400, 0),
                                                ("curriculum", "curriculum 6 stages x (150 Adam + 350 L-BFGS)            ", [5, 10, 15, 20, 25, 30], 150, 350),
                                                ("curriculum10", "curriculum 10 stages x (60 Adam + 240 L-BFGS)            ", [3, 6, 9, 12, 15, 18, 21, 24, 27, 30], 60, 240)):
            errs = []
            for seed in range(2):
                model = Net(seed); pts = points(seed); c0 = time.process_time()
                for i, b in enumerate(sched):
                    optimize(model, make_loss(model, b, pts), na, nl_stage if nl_stage else (NL if i == len(sched) - 1 else 0))
                cpu = time.process_time() - c0; e, et, up = evaluate(model, beta); errs.append(e)
                if seed == 0: keep = (et, up)
            report(key, label, errs, *keep, cpu)
        # (3) time marching: 6 slabs, each trained like the first curriculum stage, warm-started from the previous slab
        errs = []
        for seed in range(2):
            c0 = time.process_time(); nwin = 6; prev = None; U = np.zeros_like(XG)
            for k in range(nwin):
                ta, tb_ = k / nwin, (k + 1) / nwin
                model = Net(seed, t0=ta, t1=tb_)                                      # slab-local time coordinate
                if prev is not None: model.load_state_dict(prev.state_dict())       # warm start
                ic = torch.sin if prev is None else (lambda x, p=prev, ta=ta: p(x, torch.full_like(x, ta)).detach())
                optimize(model, make_loss(model, beta, points(seed + 10 * k, t0=ta, t1=tb_), ic, ta), 150, 350)
                m = (TG[0] >= ta - 1e-12) & (TG[0] <= tb_ + 1e-12)
                with torch.no_grad():
                    U[:, m] = model(torch.tensor(XG[:, m].reshape(-1, 1)), torch.tensor(TG[:, m].reshape(-1, 1))).numpy().reshape(XG[:, m].shape)
                prev = model
            cpu = time.process_time() - c0; ex = np.sin(XG - beta * TG)
            errs.append(np.linalg.norm(U - ex) / np.linalg.norm(ex))
            if seed == 0: keep = (np.linalg.norm(U - ex, axis=0) / np.linalg.norm(ex, axis=0), U.copy())
        report("marching", "time marching, 6 slabs x (150 Adam + 350 L-BFGS)         ", errs, *keep, cpu)
        # (4) periodic input embedding (hard periodic BC), vanilla training otherwise
        errs = []
        for seed in range(2):
            model = Net(seed, periodic=True); lf = make_loss(model, beta, points(seed)); c0 = time.process_time()
            optimize(model, lf, NA, NL); cpu = time.process_time() - c0; e, et, up = evaluate(model, beta); errs.append(e)
            if seed == 0: keep = (et, up); Ls = [float(v.detach()) for v in lf(parts=True)]
        report("periodic", "periodic embedding (cos x, sin x, t), vanilla training     ", errs, *keep, cpu, f", L_r {Ls[0]:.1e} L_ic {Ls[1]:.1e}")

        print("=== C. classical baseline at beta = 30: Lax-Wendroff (CFL 0.8), error at t = 1 ===")
        for N in (128, 512, 2048):
            ts = []
            for _ in range(3):
                c0 = time.process_time(); e, nt = lax_wendroff(30, N); ts.append(time.process_time() - c0)
            print(f"  N = {N:4d}, {nt:5d} steps: rel L2 error {e:.2e}, cpu {min(ts)*1e3:.1f} ms")
    if "D" in part:
        beta = 30
        print("=== D. diagnosis of the failed vanilla run (beta = 30, seed 0): residual on vs between the collocation points ===")
        model = Net(0); pts = points(0); lf = make_loss(model, beta, pts); optimize(model, lf, NA, NL)
        e, et, up = evaluate(model, beta); Lr, Li, Lb = [float(v.detach()) for v in lf(parts=True)]
        g = torch.Generator().manual_seed(999)
        xt_ = PI2 * torch.rand(20000, 1, generator=g); tt_ = torch.rand(20000, 1, generator=g)
        x = xt_.clone().requires_grad_(True); t = tt_.clone().requires_grad_(True); u = model(x, t)
        r = (torch.autograd.grad(u.sum(), t, create_graph=True)[0] + beta * torch.autograd.grad(u.sum(), x, create_graph=True)[0]).detach()
        print(f"  rel L2 error {e:.3f}; residual MSE on the 1000 training points {Lr:.2e}; on 20000 fresh points {float(r.pow(2).mean()):.2e}")
        tb = tt_.squeeze().numpy(); rb = r.squeeze().numpy() ** 2
        print("  fresh-point residual MSE by time slab: " + ", ".join(f"[{a:.1f},{a+0.2:.1f}] {rb[(tb >= a) & (tb < a + 0.2)].mean():.1e}" for a in (0.0, 0.2, 0.4, 0.6, 0.8)))
        amp = np.sqrt(2 * (up ** 2).mean(0))                 # amplitude of the predicted wave at each time (exact: 1)
        print("  wave amplitude sqrt(2<u^2>) at t = 0, 0.25, 0.5, 0.75, 1: " + ", ".join(f"{amp[i]:.3f}" for i in (0, 25, 50, 75, 100)))
        # the exact solution, for scale: residual of a wave damped as exp(-gamma t)
        gam = -math.log(max(amp[-1], 1e-12) / amp[0])
        print(f"  for scale: a wave damped like exp(-gamma t) with gamma = {gam:.2f} would have residual MSE gamma^2 <u^2> ~ {gam**2 * (up**2).mean():.2e}")
        out["diag"] = dict(amp=amp.tolist())
    print(f"part {part}: total wall {time.perf_counter()-t_all:.0f} s, cpu {time.process_time()-c_all:.0f} s")
    json.dump(out, open(JS, "w"))
