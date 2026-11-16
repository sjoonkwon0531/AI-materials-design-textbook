"""Lab 9.2-heat: 1D diffusion PINN, forward problem: soft vs hard constraints, and the Crank-Nicolson baseline.
u_t = D u_xx on (0,1)x(0,1], D=0.1, u(x,0)=sin(pi x)+0.5 sin(3 pi x), u(0,t)=u(1,t)=0.
Exact: u = exp(-D pi^2 t) sin(pi x) + 0.5 exp(-9 D pi^2 t) sin(3 pi x).
Run: python 92_heat.py            (single thread)
"""
import math, time, json, sys
import numpy as np, torch
from scipy.linalg import solve_banded

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
PI = math.pi
D_TRUE = 0.1


def u_exact(x, t, D=D_TRUE):
    return np.exp(-D * PI**2 * t) * np.sin(PI * x) + 0.5 * np.exp(-9 * D * PI**2 * t) * np.sin(3 * PI * x)


def u0_t(x):
    return torch.sin(PI * x) + 0.5 * torch.sin(3 * PI * x)


def mlp(seed, width=32, depth=3, nin=2):
    torch.manual_seed(seed)
    layers, d = [], nin
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers += [torch.nn.Linear(d, 1)]
    return torch.nn.Sequential(*layers)


def make_u(net, hard):
    if hard:   # distance-function ansatz: IC and BC hold exactly
        return lambda x, t: u0_t(x) + t * x * (1 - x) * net(torch.cat([x, t], 1))
    return lambda x, t: net(torch.cat([x, t], 1))


def residual(u, x, t, D):
    x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
    uu = u(x, t)
    u_t = torch.autograd.grad(uu.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(uu.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t - D * u_xx


# evaluation grid
xg = np.linspace(0, 1, 101); tg = np.linspace(0, 1, 101)
XG, TG = np.meshgrid(xg, tg, indexing="ij")
UG = u_exact(XG, TG)
xg_t = torch.tensor(XG.reshape(-1, 1)); tg_t = torch.tensor(TG.reshape(-1, 1))


def errors(u):
    with torch.no_grad():
        up = u(xg_t, tg_t).numpy().reshape(XG.shape)
    return np.linalg.norm(up - UG) / np.linalg.norm(UG), np.abs(up - UG).max(), up


def train(params, loss_fn, n_adam, n_lbfgs, lr=1e-3, log=None):
    opt = torch.optim.Adam(params, lr=lr)
    for it in range(n_adam):
        opt.zero_grad(); L = loss_fn(); L.backward(); opt.step()
        if log is not None and it % 50 == 0: log(it)
    if n_lbfgs > 0:
        lb = torch.optim.LBFGS(params, max_iter=n_lbfgs, tolerance_grad=1e-12, tolerance_change=1e-15,
                               history_size=50, line_search_fn="strong_wolfe")
        k = [n_adam]
        def closure():
            lb.zero_grad(); L = loss_fn(); L.backward(); k[0] += 1
            if log is not None and k[0] % 50 == 0: log(k[0])
            return L
        lb.step(closure)
    return float(loss_fn().detach())


# ------------------------------------------------------------------ classical baseline: Crank-Nicolson
def crank_nicolson(D, Nx, Nt, T=1.0, u0=None):
    x = np.linspace(0, 1, Nx + 1); dx = 1 / Nx; dt = T / Nt; lam = D * dt / dx**2
    u = (np.sin(PI * x) + 0.5 * np.sin(3 * PI * x)) if u0 is None else u0(x)
    n = Nx - 1
    ab = np.zeros((3, n)); ab[0, 1:] = -lam / 2; ab[1, :] = 1 + lam; ab[2, :-1] = -lam / 2
    out = [u.copy()]
    for _ in range(Nt):
        rhs = u[1:-1] + lam / 2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        u[1:-1] = solve_banded((1, 1), ab, rhs)
        out.append(u.copy())
    return x, np.array(out).T      # (Nx+1, Nt+1)


def cn_error(Nx, Nt):
    t0 = time.process_time(); x, U = crank_nicolson(D_TRUE, Nx, Nt); dt = time.process_time() - t0
    X, T_ = np.meshgrid(x, np.linspace(0, 1, Nt + 1), indexing="ij"); E = u_exact(X, T_)
    return np.linalg.norm(U - E) / np.linalg.norm(E), np.abs(U - E).max(), dt


if __name__ == "__main__":
    t_all = time.process_time()
    out = {}
    print("=== Part A: forward problem, soft vs hard constraints (2-32-32-32-1 tanh, 1000 collocation points) ===")
    out["A"] = {}
    for hard in (False, True):
        rel, mx, wall, hist_keep = [], [], [], None
        for seed in range(3):
            net = mlp(seed); u = make_u(net, hard)
            g = torch.Generator().manual_seed(100 + seed)
            xf = torch.rand(1000, 1, generator=g); tf = torch.rand(1000, 1, generator=g)
            x0 = torch.rand(200, 1, generator=g); tb = torch.rand(200, 1, generator=g)
            xb = torch.cat([torch.zeros(100, 1), torch.ones(100, 1)])
            def loss_fn():
                L = residual(u, xf, tf, D_TRUE).pow(2).mean()
                if not hard:
                    L = L + (u(x0, 0 * x0) - u0_t(x0)).pow(2).mean() + u(xb, tb).pow(2).mean()
                return L
            hist = []
            t0 = time.process_time()
            e_adam = None
            train(list(net.parameters()), loss_fn, 3000, 0, log=lambda it: hist.append((it, errors(u)[0])))
            e_adam = errors(u)[0]
            train(list(net.parameters()), loss_fn, 0, 1500, log=None)
            w = time.process_time() - t0
            r, m, up = errors(u)
            rel.append(r); mx.append(m); wall.append(w)
            print(f"  {'hard' if hard else 'soft'} seed {seed}: rel L2 after Adam {e_adam:.2e} -> after L-BFGS {r:.2e}, max err {m:.2e}, cpu {w:.1f} s")
            if seed == 0:
                out["A"]["hard_err" if hard else "soft_err"] = np.abs(up - UG).tolist()
        tag = "hard" if hard else "soft"
        out["A"][tag] = dict(rel=rel, mx=mx, wall=wall)
        print(f"  {tag}: rel L2 = {np.mean(rel):.2e} +/- {np.std(rel):.1e}, max = {np.mean(mx):.2e}, cpu = {np.mean(wall):.1f} s (3 seeds)")

    print("=== Crank-Nicolson baseline (Thomas solve, numpy) ===")
    out["CN"] = []
    for Nx, Nt in [(25, 25), (50, 50), (100, 100), (200, 200), (400, 400)]:
        ts = [cn_error(Nx, Nt) for _ in range(5)]
        r, m, dt = ts[0][0], ts[0][1], min(t[2] for t in ts)
        out["CN"].append((Nx, Nt, r, m, dt))
        print(f"  Nx=Nt={Nx:4d}: rel L2 {r:.2e}, max err {m:.2e}, cpu {dt*1e3:.2f} ms")

    print(f"total cpu {time.process_time()-t_all:.0f} s")
    json.dump(out, open("../ch09/data/92_heat.json", "w"))
