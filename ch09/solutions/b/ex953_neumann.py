"""Exercise 9.5.3: insulated rod u_t = D u_xx, u_x(0,t) = u_x(1,t) = 0, u(x,0) = cos(pi x), D = 0.1.
Hard constraint through a feature transform (cos pi x, t) and an output transform cos(pi x) + t N,
versus soft constraints with dde.icbc.NeumannBC. Same budget as Lab 9.15 (Adam 3000 + L-BFGS)."""
import os, sys, time
os.environ["DDE_BACKEND"] = "pytorch"
import numpy as np, torch, deepxde as dde
torch.set_num_threads(1); dde.config.set_default_float("float64")
SMOKE = len(sys.argv) > 1
D = 0.1
exact = lambda X: np.exp(-D * np.pi**2 * X[:, 1:2]) * np.cos(np.pi * X[:, 0:1])
geomtime = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1))

def pde(X, u):
    return dde.grad.jacobian(u, X, i=0, j=1) - D * dde.grad.hessian(u, X, i=0, j=0)

bc = dde.icbc.NeumannBC(geomtime, lambda X: 0.0, lambda X, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda X: np.cos(np.pi * X[:, 0:1]), lambda X, on_initial: on_initial)
xg = np.linspace(0, 1, 101); Xg, Tg = np.meshgrid(xg, xg, indexing="ij")
XT = np.stack([Xg.ravel(), Tg.ravel()], 1); Uex = exact(XT)
XB = np.concatenate([np.stack([np.zeros(101), xg], 1), np.stack([np.ones(101), xg], 1)])      # boundary points

def run(hard):
    dde.config.set_random_seed(0)
    if hard:
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=1000, solution=exact, num_test=10000)
    else:
        data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=1000, num_boundary=200, num_initial=200,
                                solution=exact, num_test=10000)
    net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    if hard:                                              # u = cos(pi x) + t N(cos(pi x), t)
        net.apply_feature_transform(lambda X: torch.cat([torch.cos(np.pi * X[:, 0:1]), X[:, 1:2]], 1))
        net.apply_output_transform(lambda X, y: torch.cos(np.pi * X[:, 0:1]) + X[:, 1:2] * y)
    model = dde.Model(data, net); rows = []
    def record(tag):
        cpu = time.process_time() - t0; pred = model.predict(XT)
        ux = model.predict(XB, operator=lambda x, y: dde.grad.jacobian(y, x, i=0, j=0))
        rows.append((tag, model.train_state.step, cpu, np.asarray(model.train_state.loss_train),
                     np.linalg.norm(pred - Uex) / np.linalg.norm(Uex), np.abs(pred - Uex).max(), np.abs(ux).max(),
                     np.abs(model.predict(XT[XT[:, 1] == 0]) - Uex[XT[:, 1] == 0]).max()))
    t0 = time.process_time()
    model.compile("adam", lr=1e-3)
    model.train(iterations=30 if SMOKE else 1000, display_every=1000, verbose=0); record("Adam 1000")
    model.train(iterations=30 if SMOKE else 2000, display_every=1000, verbose=0); record("Adam 3000")
    dde.optimizers.config.set_LBFGS_options(maxcor=50, ftol=0, gtol=1e-12, maxiter=20 if SMOKE else 2000, maxls=50)
    model.compile("L-BFGS"); model.train(display_every=1000, verbose=0); record("+ L-BFGS")
    return rows

for hard in (False, True):
    rows = run(hard)
    print(f"\n{'hard (feature + output transform)' if hard else 'soft (PDE + NeumannBC + IC losses)'}: FNN 2-32-32-32-1 tanh, {1000 if hard else 1400} training points")
    print("  stage        steps   CPU (s)   loss terms                       rel. L2 (101x101)   max|err|    max|u_x| on x=0,1   max IC error")
    for tag, st, cpu, lt, rel, emax, uxm, icm in rows:
        print(f"  {tag:<11s} {st:6d}   {cpu:6.1f}   {' / '.join(f'{v:.1e}' for v in lt):<30s}   {rel:.2e}            {emax:.2e}    {uxm:.2e}            {icm:.2e}", flush=True)
