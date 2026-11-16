"""Lab 9.2-inverse: identify the diffusion coefficient D from sparse noisy data with a PINN.
Same problem as 92_heat.py (u_t = D u_xx, D_true = 0.1). D enters the loss as a trainable parameter log D.
B1: 40 scattered noisy points, IC and BC known (hard ansatz).  B2: 5 sensors x 16 times, IC and BC unknown.
Classical baselines: CN forward solver inside a scalar least squares (B1), sine-series least squares (B2).
"""
import math, time, json, importlib
import numpy as np, torch
from scipy.optimize import minimize_scalar, least_squares
H = importlib.import_module("92_heat")          # reuse mlp, make_u, residual, train, crank_nicolson, u_exact
mlp, make_u, residual, train, crank_nicolson, u_exact = H.mlp, H.make_u, H.residual, H.train, H.crank_nicolson, H.u_exact
PI, D_TRUE = math.pi, H.D_TRUE
torch.set_num_threads(1)

if __name__ == "__main__":
    t_all = time.process_time(); out = {}
    print("=== inverse problem: identify D from sparse noisy data (noise sigma = 0.02, initial guess D = 0.5) ===")
    sigma = 0.02
    def run_inverse(case, seed, n_adam=3000, n_lbfgs=500, verbose=False):
        rng = np.random.default_rng(seed)
        if case == "B1":      # 40 scattered points; IC and BC known (hard ansatz)
            xd = rng.uniform(0.05, 0.95, 40); td = rng.uniform(0.05, 1.0, 40)
        else:                 # B2: 5 sensors x 16 times, t in [0.1,1]; IC and BC NOT given to the model
            xs = np.array([0.15, 0.3, 0.5, 0.7, 0.85]); ts = np.linspace(0.1, 1.0, 16)
            xd, td = [a.ravel() for a in np.meshgrid(xs, ts, indexing="ij")]
        yd = u_exact(xd, td) + sigma * rng.standard_normal(xd.size)
        xd_t = torch.tensor(xd[:, None]); td_t = torch.tensor(td[:, None]); yd_t = torch.tensor(yd[:, None])
        net = mlp(seed); u = make_u(net, hard=(case == "B1"))
        logD = torch.nn.Parameter(torch.tensor(math.log(0.5)))       # initial guess D = 0.5 (5x too large)
        g = torch.Generator().manual_seed(200 + seed)
        if case == "B1":
            xf = torch.rand(1000, 1, generator=g); tf = torch.rand(1000, 1, generator=g)
        else:                 # collocation only inside the convex hull of the sensors
            xf = 0.15 + 0.7 * torch.rand(1000, 1, generator=g); tf = 0.1 + 0.9 * torch.rand(1000, 1, generator=g)
        def loss_fn():
            return residual(u, xf, tf, torch.exp(logD)).pow(2).mean() + 10.0 * (u(xd_t, td_t) - yd_t).pow(2).mean()
        traj = []
        t0 = time.process_time()
        train(list(net.parameters()) + [logD], loss_fn, n_adam, n_lbfgs, lr=2e-3,
              log=lambda it: traj.append((it, float(torch.exp(logD)))))
        w = time.process_time() - t0
        with torch.no_grad():
            rmse_fit = float((u(xd_t, td_t) - yd_t).pow(2).mean().sqrt())
        return float(torch.exp(logD)), w, traj, (xd, td, yd), rmse_fit

    def classical_B1(seed):
        """Scalar least squares in D using the CN solver as forward model (needs IC and BC)."""
        rng = np.random.default_rng(seed)
        xd = rng.uniform(0.05, 0.95, 40); td = rng.uniform(0.05, 1.0, 40)
        yd = u_exact(xd, td) + sigma * rng.standard_normal(40)
        from scipy.interpolate import RegularGridInterpolator
        def sse(D):
            x, U = crank_nicolson(D, 100, 100)
            f = RegularGridInterpolator((x, np.linspace(0, 1, 101)), U)
            return np.sum((f(np.c_[xd, td]) - yd) ** 2)
        t0 = time.process_time(); r = minimize_scalar(sse, bounds=(0.01, 1.0), method="bounded", options=dict(xatol=1e-6))
        return r.x, time.process_time() - t0, r.nfev

    def classical_B2(seed):
        """Without IC: fit D and the amplitudes of 4 sine modes (assumes the homogeneous Dirichlet BC is known)."""
        rng = np.random.default_rng(seed)
        xs = np.array([0.15, 0.3, 0.5, 0.7, 0.85]); ts = np.linspace(0.1, 1.0, 16)
        xd, td = [a.ravel() for a in np.meshgrid(xs, ts, indexing="ij")]
        yd = u_exact(xd, td) + sigma * rng.standard_normal(xd.size)
        ks = np.arange(1, 5)
        def res(p):
            D, a = math.exp(p[0]), p[1:]
            return (np.exp(-D * (ks[None] * PI) ** 2 * td[:, None]) * np.sin(ks[None] * PI * xd[:, None])) @ a - yd
        t0 = time.process_time(); r = least_squares(res, np.r_[math.log(0.5), np.zeros(4)])
        return math.exp(r.x[0]), time.process_time() - t0

    out["B"] = {}
    for case in ("B1", "B2"):
        Ds, ws, rm = [], [], []
        for seed in range(5):
            Dhat, w, traj, data, rmse_fit = run_inverse(case, seed)
            Ds.append(Dhat); ws.append(w); rm.append(rmse_fit)
            if seed == 0:
                out["B"][case + "_traj"] = traj; out["B"][case + "_data"] = [d.tolist() for d in data]
            print(f"  {case} seed {seed}: D_hat = {Dhat:.4f} (true 0.1000), data RMSE {rmse_fit:.4f} (noise sigma {sigma}), cpu {w:.1f} s")
        out["B"][case] = dict(D=Ds, wall=ws)
        print(f"  {case} PINN : D = {np.mean(Ds):.4f} +/- {np.std(Ds):.4f} (5 noise/init seeds), |bias| {abs(np.mean(Ds)-D_TRUE)/D_TRUE*100:.1f} %, cpu {np.mean(ws):.1f} s")
        if case == "B1":
            cl = [classical_B1(s) for s in range(5)]
            Dc = [c[0] for c in cl]
            print(f"  B1 CN + scalar least squares: D = {np.mean(Dc):.4f} +/- {np.std(Dc):.4f}, cpu {np.mean([c[1] for c in cl])*1e3:.0f} ms ({cl[0][2]} forward solves)")
            out["B"]["B1_classical"] = Dc
        else:
            cl = [classical_B2(s) for s in range(5)]
            Dc = [c[0] for c in cl]
            print(f"  B2 sine-series least squares (BC assumed known): D = {np.mean(Dc):.4f} +/- {np.std(Dc):.4f}, cpu {np.mean([c[1] for c in cl])*1e3:.0f} ms")
            out["B"]["B2_classical"] = Dc
    print(f"total cpu {time.process_time()-t_all:.0f} s")
    json.dump(out, open("../ch09/data/92_inverse.json", "w"))
