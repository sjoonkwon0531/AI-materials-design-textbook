"""Lab 9.2-spectral: spectral bias, Fourier features and NTK loss balancing on a two-scale Poisson problem.
-u'' = f on (0,1), u(0)=u(1)=0, exact u = sin(2 pi x) + 0.1 sin(50 pi x).
A. plain regression of u: which frequency is learned first?
C. empirical NTK blocks K_uu (boundary) and K_rr (residual): traces and top eigenvalues.
B. PINN: tanh MLP vs Fourier-feature network (sigma = 10); Adam with equal or NTK-trace weights, Adam + L-BFGS,
   and Levenberg-Marquardt in kernel (NTK) form.
Baseline: second-order finite differences.
"""
import math, time, json, sys
import numpy as np, torch
from scipy.linalg import solve_banded

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
PI = math.pi
u_ex = lambda x: torch.sin(2 * PI * x) + 0.1 * torch.sin(50 * PI * x)
f_rhs = lambda x: (2 * PI) ** 2 * torch.sin(2 * PI * x) + 0.1 * (50 * PI) ** 2 * torch.sin(50 * PI * x)
xe = torch.linspace(0, 1, 2001)[:, None]; ue = u_ex(xe)
NR = 256
xr = torch.linspace(0, 1, NR + 2)[1:-1, None]            # interior collocation points
xb = torch.tensor([[0.0], [1.0]])


class Net(torch.nn.Module):
    def __init__(s, seed, sigma=None, m=64, width=64, hard=False):
        super().__init__()
        g = torch.Generator().manual_seed(1000 + seed)
        s.B = None if sigma is None else sigma * torch.randn(1, m, generator=g)     # fixed random frequencies
        torch.manual_seed(seed)
        nin = 1 if sigma is None else 2 * m
        s.net = torch.nn.Sequential(torch.nn.Linear(nin, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(),
                                    torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
        s.hard = hard
    def forward(s, x):
        z = x if s.B is None else torch.cat([torch.cos(2 * PI * x @ s.B), torch.sin(2 * PI * x @ s.B)], 1)
        y = s.net(z)
        return x * (1 - x) * y if s.hard else y


def resid(model, x):
    x = x.clone().requires_grad_(True); u = model(x)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    return -uxx - f_rhs(x)


def ntk_traces(model, full=False):
    """Tr K_uu and Tr K_rr of the empirical NTK (K = J J^T, J = d(outputs)/d(theta))."""
    ps = [p for p in model.parameters()]
    def jac(vec):
        rows = []
        for v in vec:
            g = torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1) for gi, p in zip(g, ps)]))
        return torch.stack(rows)
    Ju = jac(model(xb).reshape(-1)); Jr = jac(resid(model, xr).reshape(-1))
    if full:
        Kuu, Krr = Ju @ Ju.T, Jr @ Jr.T
        return float(torch.trace(Kuu)), float(torch.trace(Krr)), float(torch.linalg.eigvalsh(Kuu)[-1]), torch.linalg.eigvalsh(Krr).numpy()
    return float((Ju ** 2).sum()), float((Jr ** 2).sum())


def rel_err(model):
    with torch.no_grad():
        return float(torch.linalg.norm(model(xe) - ue) / torch.linalg.norm(ue))


def mode_amps(model):
    """Sine-series coefficients of the prediction at k=2 (target 1) and k=50 (target 0.1)."""
    with torch.no_grad():
        up = model(xe)
        a = lambda k: float(2 * torch.trapezoid((up * torch.sin(k * PI * xe)).squeeze(), xe.squeeze()))
    return a(2), a(50)


def train_pinn(model, weighting, iters, n_lbfgs=0, lr=1e-3, log_every=250):
    """Adam (optionally followed by L-BFGS). History rows: (cpu seconds, rel L2 error, L_r, L_b, lam_b/lam_r)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9 ** (1 / 1000))        # x0.9 every 1000 it
    lam_b, lam_r, hist, c0 = 1.0, 1.0, [], time.process_time()
    def losses():
        return resid(model, xr).pow(2).mean(), model(xb).pow(2).mean()
    def log():
        Lr, Lb = losses(); hist.append((time.process_time() - c0, rel_err(model), float(Lr.detach()), float(Lb.detach()), lam_b / lam_r))
    for it in range(iters):
        if weighting == "ntk" and it % 1000 == 0:
            tu, tr = ntk_traces(model); lam_b, lam_r = (tu + tr) / tu, (tu + tr) / tr          # NTK-trace weights
        if it % log_every == 0: log()
        opt.zero_grad(); Lr, Lb = losses()
        (lam_r * Lr + (0.0 if model.hard else lam_b * Lb)).backward(); opt.step(); sched.step()
    log()
    if n_lbfgs:
        lb = torch.optim.LBFGS(model.parameters(), max_iter=n_lbfgs, history_size=100, tolerance_grad=1e-14,
                               tolerance_change=1e-16, line_search_fn="strong_wolfe")
        k = [0]
        def closure():
            lb.zero_grad(); Lr, Lb = losses(); L = lam_r * Lr + (0.0 if model.hard else lam_b * Lb); L.backward()
            k[0] += 1
            if k[0] % log_every == 0: log()
            return L
        lb.step(closure); log()
    return hist


def train_lm(model, iters, mu=1e-2):
    """Levenberg-Marquardt in kernel form: d theta = -J^T (K + mu I)^{-1} r with K = J J^T the empirical NTK."""
    from torch.func import functional_call, grad, vmap, jacrev
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for p in model.parameters()]; sizes = [p.numel() for p in model.parameters()]
    def unflat(v):
        d, i = {}, 0
        for n, sh, k in zip(names, shapes, sizes):
            d[n] = v[i:i + k].reshape(sh); i += k
        return d
    u1 = lambda v, x: functional_call(model, unflat(v), (x.reshape(1, 1),)).squeeze()       # scalar x -> scalar u
    uxx = grad(grad(u1, argnums=1), argnums=1)
    xr1, xb1 = xr.squeeze(1), xb.squeeze(1)
    def resvec(v):                                   # stacked residual vector [r / sqrt(N_r); u_b / sqrt(N_b)]
        r = -vmap(lambda x: uxx(v, x))(xr1) - f_rhs(xr1)
        return torch.cat([r / NR ** 0.5, vmap(lambda x: u1(v, x))(xb1) / 2 ** 0.5])
    def write(v):
        i = 0
        with torch.no_grad():
            for p, k in zip(model.parameters(), sizes):
                p.copy_(v[i:i + k].reshape(p.shape)); i += k
    v = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    r = resvec(v); L = float(r @ r); hist, c0 = [], time.process_time()
    for it in range(iters):
        if it % 5 == 0:
            write(v); hist.append((time.process_time() - c0, rel_err(model), L, 0.0, 1.0))
        J = jacrev(resvec)(v); K = J @ J.T          # (N_r + 2) x (N_r + 2)
        while mu < 1e12:
            dv = -J.T @ torch.linalg.solve(K + mu * torch.eye(K.shape[0]), r)
            r_new = resvec(v + dv); L_new = float(r_new @ r_new)
            if L_new < L:
                v, r, L, mu = v + dv, r_new, L_new, max(mu / 3, 1e-12); break
            mu *= 4
    write(v); hist.append((time.process_time() - c0, rel_err(model), L, 0.0, 1.0))
    return hist


if __name__ == "__main__":
    t_all = time.perf_counter(); c_all = time.process_time(); out = {}
    quick = len(sys.argv) > 1
    print("=== A. regression of u(x) (no PDE): amplitude of the k=2 and k=50 sine modes (targets 1 and 0.1) ===")
    out["A"] = {}
    for name, sigma in (("tanh MLP", None), ("Fourier features sigma=10", 10.0)):
        model = Net(0, sigma); opt = torch.optim.Adam(model.parameters(), lr=1e-3); rec = []
        xt = torch.linspace(0, 1, 512)[:, None]; yt = u_ex(xt)
        for it in range((300 if quick else 5000) + 1):
            if it in (0, 100, 300, 1000, 2000, 5000):
                a2, a50 = mode_amps(model); rec.append((it, a2, a50, rel_err(model)))
            opt.zero_grad(); (model(xt) - yt).pow(2).mean().backward(); opt.step()
        out["A"][name] = rec
        print(f"  {name:26s} " + " | ".join(f"it {r[0]:4d}: a2 {r[1]:.3f} a50 {r[2]:+.4f}" for r in rec[1:]))
        print(f"  {'':26s} final rel L2 error {rec[-1][3]:.2e}")

    print("=== C. empirical NTK at initialization (256 residual points, 2 boundary points) ===")
    out["C"] = {}
    for name, sigma in (("tanh MLP", None), ("FF sigma=1", 1.0), ("FF sigma=10", 10.0), ("FF sigma=50", 50.0)):
        tu, tr, lu, er = ntk_traces(Net(0, sigma), full=True)
        out["C"][name] = dict(tr_uu=tu, tr_rr=tr, eig_rr=er.tolist(), lam_uu=lu)
        print(f"  {name:12s}: Tr K_uu/N_b = {tu/2:9.3e}   Tr K_rr/N_r = {tr/NR:9.3e}   ratio = {tr/NR/(tu/2):8.2e}   lam_max(K_rr) = {er[-1]:.2e}")

    print(f"=== B. PINN, N_r = {NR}; rel L2 error on 2001 points; cpu = process CPU seconds ===")
    out["B"] = {}
    NAD = 300 if quick else 3000
    runs = [("tanh MLP, Adam", None, "adam", "equal", (0,)), ("FF sigma=10, Adam, equal weights", 10.0, "adam", "equal", (0,)),
            ("FF sigma=10, Adam, NTK weights", 10.0, "adam", "ntk", (0,)), ("FF sigma=10, Adam + L-BFGS", 10.0, "lbfgs", "equal", (0, 1)),
            ("FF sigma=10, Levenberg-Marquardt", 10.0, "lm", "equal", (0, 1)), ("tanh MLP, Levenberg-Marquardt", None, "lm", "equal", (0,))]
    for name, sigma, method, wt, seeds in runs:
        errs = []
        for seed in seeds:
            model = Net(seed, sigma)
            if method == "adam": hist = train_pinn(model, wt, NAD)
            elif method == "lbfgs": hist = train_pinn(model, wt, NAD // 6, n_lbfgs=NAD - NAD // 6)
            else: hist = train_lm(model, 10 if quick else 60)
            errs.append(hist[-1][1])
            if seed == seeds[0]:
                a2, a50 = mode_amps(model)
                with torch.no_grad():
                    ub = model(xb).squeeze().tolist(); up = model(xe).squeeze().numpy()
                Lr = float(resid(model, xr).pow(2).mean().detach())
                out["B"][name] = dict(hist=hist, pred=up[::2].tolist())
                extra = f"a2 {a2:.3f} a50 {a50:.4f} u(0) {ub[0]:+.1e} u(1) {ub[1]:+.1e} L_r {Lr:.1e} lam_b/lam_r {hist[-1][4]:.1e} cpu {hist[-1][0]:.0f} s"
        out["B"][name]["errs"] = errs
        print(f"  {name:33s}: rel L2 = {', '.join(f'{e:.2e}' for e in errs)} | seed {seeds[0]}: {extra}")

    print("=== baseline: second-order finite differences (Thomas algorithm) ===")
    for n in (256, 1024, 4096):
        ts = []
        for _ in range(20):
            t0 = time.process_time()
            x = np.linspace(0, 1, n + 1); h = 1 / n
            fx = (2 * PI) ** 2 * np.sin(2 * PI * x) + 0.1 * (50 * PI) ** 2 * np.sin(50 * PI * x)
            ab = np.zeros((3, n - 1)); ab[0, 1:] = -1; ab[1] = 2; ab[2, :-1] = -1
            u = np.r_[0, solve_banded((1, 1), ab, h * h * fx[1:-1]), 0]
            ts.append(time.process_time() - t0)
        ex = np.sin(2 * PI * x) + 0.1 * np.sin(50 * PI * x)
        print(f"  N = {n:5d}: rel L2 error {np.linalg.norm(u-ex)/np.linalg.norm(ex):.2e}, cpu {min(ts)*1e3:.3f} ms")
    print(f"total wall {time.perf_counter()-t_all:.0f} s, cpu {time.process_time()-c_all:.0f} s")
    if not quick: json.dump(out, open("../../ch09/data/92_spectral.json", "w"))
