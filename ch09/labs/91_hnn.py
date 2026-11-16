"""Lab 9.1: learn the right-hand side of a pendulum from noisy trajectory snapshots.
Neural ODE (plain MLP vector field) vs Hamiltonian NN (scalar H_theta, symplectic gradient).
True system: H = p^2/2 + (1 - cos q),  dq/dt = p, dp/dt = -sin q.
Training is "discretize-then-optimize": backprop through K RK4 steps (no derivative labels).
"""
import math, time, json
import numpy as np, torch
from scipy.integrate import solve_ivp

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

H_true = lambda q, p: 0.5 * p**2 + (1 - np.cos(q))
f_true = lambda t, y: [y[1], -np.sin(y[0])]

# ---------------- data: 40 librating trajectories, 21 snapshots each (dt = 0.2), noise sigma = 0.02
DT, NSNAP, NTRAJ, SIGMA, K = 0.2, 21, 40, 0.02, 5
rng = np.random.default_rng(0)
trajs = []
while len(trajs) < NTRAJ:
    q0, p0 = rng.uniform(-2.2, 2.2), rng.uniform(-1.5, 1.5)
    if not (0.05 < H_true(q0, p0) < 1.8):          # stay below the separatrix (E = 2)
        continue
    s = solve_ivp(f_true, [0, DT * (NSNAP - 1)], [q0, p0], t_eval=np.arange(NSNAP) * DT, rtol=1e-10, atol=1e-12)
    trajs.append(s.y.T)
clean = np.array(trajs)                               # (40, 21, 2)
noisy = clean + SIGMA * rng.standard_normal(clean.shape)
# windows of K+1 consecutive snapshots
idx = np.arange(NSNAP - K)
X0 = torch.tensor(noisy[:, idx].reshape(-1, 2))                                  # (40*16, 2)
Y = torch.tensor(np.stack([noisy[:, idx + j] for j in range(1, K + 1)], 2).reshape(-1, K, 2))


class MLPField(torch.nn.Module):
    """Neural ODE: dx/dt = f_theta(x), no structure."""
    def __init__(s, w=64):
        super().__init__()
        s.net = torch.nn.Sequential(torch.nn.Linear(2, w), torch.nn.Tanh(), torch.nn.Linear(w, w), torch.nn.Tanh(), torch.nn.Linear(w, 2))
    def forward(s, x):
        return s.net(x)


class HNN(torch.nn.Module):
    """dx/dt = J grad H_theta(x): the vector field is a symplectic gradient by construction."""
    def __init__(s, w=64):
        super().__init__()
        s.net = torch.nn.Sequential(torch.nn.Linear(2, w), torch.nn.Tanh(), torch.nn.Linear(w, w), torch.nn.Tanh(), torch.nn.Linear(w, 1))
    def H(s, x):
        return s.net(x).squeeze(-1)
    def forward(s, x):
        with torch.enable_grad():
            if not x.requires_grad: x = x.clone().requires_grad_(True)
            dH = torch.autograd.grad(s.H(x).sum(), x, create_graph=True)[0]
        return torch.stack([dH[:, 1], -dH[:, 0]], 1)            # (dH/dp, -dH/dq)


def rk4(f, x, h):
    k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def fit(model, seed, iters=1200):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        opt.zero_grad(); x = X0; loss = 0.0
        for j in range(K):                       # roll out K snapshots, one RK4 step (h = 0.2) each
            x = rk4(model, x, DT)
            loss = loss + (x - Y[:, j]).pow(2).mean()
        (loss / K).backward(); opt.step(); sched.step()
    return float(loss.detach() / K)


def rollout(model, x0, T, h=0.02):
    x = torch.tensor([x0]); out = [x.detach().numpy()[0].copy()]
    for _ in range(int(round(T / h))):
        x = rk4(model, x, h).detach()
        out.append(x.numpy()[0].copy())
    return np.array(out)


if __name__ == "__main__":
    t_all = time.perf_counter()
    x0 = [2.0, 0.0]; T = 200.0; h = 0.02; E0 = H_true(*x0)                 # test orbit, E0 = 1.416
    tt = np.arange(int(round(T / h)) + 1) * h
    ref = solve_ivp(f_true, [0, T], x0, t_eval=tt, rtol=1e-11, atol=1e-12).y.T
    print(f"data: {NTRAJ} trajectories x {NSNAP} snapshots (dt={DT}), noise sigma={SIGMA}; {X0.shape[0]} windows of K={K} steps")
    print(f"test orbit: q0=2.0, p0=0, E0={E0:.4f}, period {2*math.pi/ (math.pi/2/ __import__('scipy.special').special.ellipk(math.sin(1.0)**2)):.3f}, T={T:.0f}")
    res = {}
    for name, cls in (("NeuralODE-MLP", MLPField), ("HNN", HNN)):
        dE, e20, eT, fe, tr, Hd = [], [], [], [], [], []
        for seed in range(3):
            torch.manual_seed(seed); model = cls()
            t0 = time.perf_counter(); L = fit(model, seed); w = time.perf_counter() - t0
            traj = rollout(model, x0, T, h)
            E = H_true(traj[:, 0], traj[:, 1])
            # vector-field error on a grid inside the training region
            qq, pp = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-1.5, 1.5, 31)); m = H_true(qq, pp) < 1.8
            xg = torch.tensor(np.c_[qq[m], pp[m]]); fg = model(xg).detach().numpy()
            ft = np.c_[pp[m], -np.sin(qq[m])]
            fe.append(np.sqrt(((fg - ft) ** 2).sum(1).mean())); tr.append(w)
            dE.append(E[-1] - E0); e20.append(np.linalg.norm(traj[1000] - ref[1000])); eT.append(np.abs(E - E0).max())
            line = f"  {name} seed {seed}: train loss {L:.2e}, field RMSE {fe[-1]:.4f}, |x-x_ref|(t=20) {e20[-1]:.3f}, E(T)-E0 {dE[-1]:+.4f}, max|E-E0| {eT[-1]:.4f}, train {w:.0f} s"
            if name == "HNN":
                Hth = model.H(torch.tensor(traj)).detach().numpy(); Hd.append(np.abs(Hth - Hth[0]).max())
                line += f", max|H_theta drift| {Hd[-1]:.1e}"
            print(line)
            if seed == 0:
                res[name] = dict(traj=traj[::5].tolist(), E=E[::5].tolist())
        print(f"  {name}: E(T)-E0 = {np.mean(dE):+.4f} +/- {np.std(dE):.4f}; max|E-E0| = {np.mean(eT):.4f} +/- {np.std(eT):.4f}; field RMSE {np.mean(fe):.4f} +/- {np.std(fe):.4f}; train {np.mean(tr):.0f} s")
    # reference: energy error of plain RK4 on the TRUE field with the same step (integrator error only)
    x = np.array(x0); Emax = 0.0
    fnp = lambda y: np.array([y[1], -np.sin(y[0])])
    for _ in range(int(round(T / h))):
        k1 = fnp(x); k2 = fnp(x + 0.5 * h * k1); k3 = fnp(x + 0.5 * h * k2); k4 = fnp(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); Emax = max(Emax, abs(H_true(*x) - E0))
    print(f"  true field + RK4 (h={h}): max|E-E0| = {Emax:.1e}  (integrator error is negligible)")
    res["ref"] = ref[::5].tolist(); res["t"] = tt[::5].tolist(); res["train_clean"] = clean.tolist(); res["train_noisy"] = noisy.tolist()
    json.dump(res, open("../ch09/data/91_hnn.json", "w"))
    print(f"total wall {time.perf_counter()-t_all:.0f} s")
