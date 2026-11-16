"""Lab 9.5 (a): DeepXDE (PyTorch backend) PINN for the 1D heat equation
    u_t = D u_xx,  D = 0.1,  x in (0,1), t in (0,1],  u(0,t) = u(1,t) = 0,  u(x,0) = sin(pi x),
exact solution u = exp(-D pi^2 t) sin(pi x)  (the problem of Ch.1 exercise 1.8.D).
Two PINNs (soft constraints; hard constraints through an output transform) are compared with
the Crank-Nicolson scheme of Ch.2. float64, one thread, seed 0."""
import os, time
os.environ["DDE_BACKEND"] = "pytorch"
import numpy as np
import torch
import deepxde as dde
from scipy.linalg import solve_banded

torch.set_num_threads(1)
dde.config.set_default_float("float64")
D = 0.1
exact = lambda X: np.exp(-D * np.pi**2 * X[:, 1:2]) * np.sin(np.pi * X[:, 0:1])

# ---------------------------------------------------------------- classical baseline: Crank-Nicolson
def crank_nicolson(nx, nt):
    """Return u on the (nx+1) x (nt+1) grid; tridiagonal solve with scipy.linalg.solve_banded."""
    x = np.linspace(0, 1, nx + 1); dt = 1.0 / nt; r = D * dt / (1.0 / nx) ** 2
    u = np.sin(np.pi * x); out = np.empty((nt + 1, nx + 1)); out[0] = u
    n = nx - 1
    ab = np.zeros((3, n)); ab[0, 1:] = -r / 2; ab[1] = 1 + r; ab[2, :-1] = -r / 2
    for k in range(nt):
        rhs = (1 - r) * u[1:-1] + (r / 2) * (u[2:] + u[:-2])
        u = np.concatenate([[0.0], solve_banded((1, 1), ab, rhs), [0.0]])
        out[k + 1] = u
    return x, out

def best_time(f, rep=30):
    """Smallest CPU time of one call (the machine is shared, so wall time is not meaningful)."""
    best = np.inf
    for _ in range(rep):
        t0 = time.process_time(); f(); best = min(best, time.process_time() - t0)
    return best

print("Crank-Nicolson (nx = nt = n):   n   max|err| at t=1   rel. L2 on its own grid   CPU (ms)")
cn_rows = []
for n in (25, 50, 100, 200, 400, 800):
    x, U = crank_nicolson(n, n)
    T_, X_ = np.meshgrid(np.linspace(0, 1, n + 1), x, indexing="ij")
    Uex = np.exp(-D * np.pi**2 * T_) * np.sin(np.pi * X_)
    e_inf = np.abs(U[-1] - Uex[-1]).max(); e_l2 = np.linalg.norm(U - Uex) / np.linalg.norm(Uex)
    ms = 1e3 * best_time(lambda: crank_nicolson(n, n))
    cn_rows.append((n, e_inf, e_l2, ms))
    print(f"                              {n:4d}   {e_inf:.2e}          {e_l2:.2e}                {ms:8.2f}")

# ---------------------------------------------------------------- the PINN in DeepXDE
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(X, u):                                        # X[:, 0] = x, X[:, 1] = t
    u_t = dde.grad.jacobian(u, X, i=0, j=1)
    u_xx = dde.grad.hessian(u, X, i=0, j=0)
    return u_t - D * u_xx

bc = dde.icbc.DirichletBC(geomtime, lambda X: 0.0, lambda X, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda X: np.sin(np.pi * X[:, 0:1]), lambda X, on_initial: on_initial)

xg = np.linspace(0, 1, 101)
Xg, Tg = np.meshgrid(xg, xg, indexing="ij")
XT = np.stack([Xg.ravel(), Tg.ravel()], 1)
Uex = exact(XT)

def run(hard):
    dde.config.set_random_seed(0)
    if hard:
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=1000, solution=exact, num_test=10000)
    else:
        data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=1000, num_boundary=200,
                                num_initial=200, solution=exact, num_test=10000)
    net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    if hard:                                          # u = sin(pi x) + t x (1 - x) N(x, t): IC and BC exact
        net.apply_output_transform(
            lambda X, y: torch.sin(np.pi * X[:, 0:1]) + X[:, 1:2] * X[:, 0:1] * (1 - X[:, 0:1]) * y)
    model = dde.Model(data, net)
    rows = []
    def record(tag, steps):
        cpu = time.process_time() - t0                 # CPU time: the machine is shared, wall time is not meaningful
        pred = model.predict(XT)
        rel = np.linalg.norm(pred - Uex) / np.linalg.norm(Uex)
        rows.append((tag, steps, cpu, rel, np.abs(pred - Uex).max(), np.asarray(model.train_state.loss_train)))
    t0 = time.process_time()
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    h1, _ = model.train(iterations=1000, display_every=100, verbose=0)
    record("Adam 1000", 1000)
    h2, _ = model.train(iterations=2000, display_every=100, verbose=0)
    record("Adam 3000", 3000)
    dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=0, gtol=1e-12, maxiter=2000, maxls=50)
    model.compile("L-BFGS", metrics=["l2 relative error"])
    h3, _ = model.train(display_every=1000, verbose=0)
    record("+ L-BFGS", model.train_state.step)
    pred = model.predict(XT).reshape(101, 101)
    steps = np.asarray(h3.steps)                       # the loss history is cumulative over train() calls
    loss = np.asarray(h3.loss_train).sum(1)
    met = np.asarray(h3.metrics_test).ravel()
    torch.save(net.state_dict(), f"../data/95_dde_heat_{'hard' if hard else 'soft'}.pt")
    return rows, pred, steps, loss, met

results = {}
for hard in (False, True):
    name = "hard constraints (output transform)" if hard else "soft constraints (PDE + BC + IC losses)"
    rows, pred, steps, loss, met = run(hard)
    results["hard" if hard else "soft"] = (rows, pred, steps, loss, met)
    print(f"\nDeepXDE PINN, {name}: FNN 2-32-32-32-1 tanh, {1000 + (0 if hard else 400)} training points")
    print("  stage        steps   CPU (s)    loss terms                         rel. L2 (101x101)   max|err|")
    for tag, st, wall, rel, emax, lt in rows:
        print(f"  {tag:<11s} {st:6d}   {wall:7.1f}   {' / '.join(f'{v:.1e}' for v in lt):<32s}   {rel:.2e}            {emax:.2e}")
    U = pred
    print(f"  u(0.5, 0.5): PINN {U[50, 50]:.6f}, exact {np.exp(-D * np.pi**2 * 0.5):.6f};"
          f"  u(0.5, 1): PINN {U[50, 100]:.6f}, exact {np.exp(-D * np.pi**2):.6f}")
    print(f"  max|err| on the line t = 1: {np.abs(U[:, 100] - np.exp(-D * np.pi**2) * np.sin(np.pi * xg)).max():.2e}")

np.savez("../data/95_dde_heat_results.npz", cn=np.array(cn_rows),
         **{f"{k}_{n}": v for k, r in results.items() for n, v in
            zip(("pred", "steps", "loss", "metric"), r[1:])},
         **{f"{k}_summary": np.array([[st, wall, rel, emax] for _, st, wall, rel, emax, _ in r[0]])
            for k, r in results.items()})
