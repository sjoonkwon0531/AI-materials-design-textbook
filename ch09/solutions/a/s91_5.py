"""Ex 9.1.5: damped pendulum; NeuralODE-MLP vs HNN vs dissipative HNN (J - R) grad H, R = L L^T constant."""
import importlib.util, sys, time, math
import numpy as np, torch
from scipy.integrate import solve_ivp
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("hnn", "../../../ch09/labs/91_hnn.py")
Lb = importlib.util.module_from_spec(spec); spec.loader.exec_module(Lb)
torch.set_default_dtype(torch.float64)
H_true = Lb.H_true; GAM = 0.05
f_damp = lambda t, y: [y[1], -np.sin(y[0]) - GAM * y[1]]
DT, NSNAP, NTRAJ, SIGMA, K = 0.2, 21, 40, 0.02, 5
rng = np.random.default_rng(0); trajs = []
while len(trajs) < NTRAJ:
    q0, p0 = rng.uniform(-2.2, 2.2), rng.uniform(-1.5, 1.5)
    if not (0.05 < H_true(q0, p0) < 1.8): continue
    s = solve_ivp(f_damp, [0, DT * (NSNAP - 1)], [q0, p0], t_eval=np.arange(NSNAP) * DT, rtol=1e-10, atol=1e-12)
    trajs.append(s.y.T)
clean = np.array(trajs); noisy = clean + SIGMA * rng.standard_normal(clean.shape)
idx = np.arange(NSNAP - K)
Lb.X0 = torch.tensor(noisy[:, idx].reshape(-1, 2))
Lb.Y = torch.tensor(np.stack([noisy[:, idx + j] for j in range(1, K + 1)], 2).reshape(-1, K, 2))

class DHNN(Lb.HNN):
    """dx/dt = (J - R) grad H_theta, R = L L^T (constant, learnable, positive semidefinite)."""
    def __init__(s, w=64):
        super().__init__(w); s.Lp = torch.nn.Parameter(0.1 * torch.eye(2))
    def R(s):
        Lt = torch.tril(s.Lp); return Lt @ Lt.T
    def forward(s, x):
        with torch.enable_grad():
            if not x.requires_grad: x = x.clone().requires_grad_(True)
            dH = torch.autograd.grad(s.H(x).sum(), x, create_graph=True)[0]
        return torch.stack([dH[:, 1], -dH[:, 0]], 1) - dH @ s.R().T

x0 = [2.0, 0.0]; T = 60.0; h = 0.02; n = int(round(T / h)); tt = np.arange(n + 1) * h
ref = solve_ivp(f_damp, [0, T], x0, t_eval=tt, rtol=1e-11, atol=1e-12).y.T
Eref = H_true(ref[:, 0], ref[:, 1])
print(f"damped pendulum gamma={GAM}: reference E(0)={Eref[0]:.4f}, E(20)={Eref[1000]:.4f}, E(40)={Eref[2000]:.4f}, E(60)={Eref[-1]:.4f}; amplitude max|q| on [50,60] = {np.abs(ref[2500:,0]).max():.3f}")
which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mlp", "hnn", "dhnn"]
seeds = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
out = {}
for name, cls in (("mlp", Lb.MLPField), ("hnn", Lb.HNN), ("dhnn", DHNN)):
    if name not in which: continue
    for seed in seeds:
        torch.manual_seed(seed); m = cls(); c0 = time.process_time(); loss = Lb.fit(m, seed); cpu = time.process_time() - c0
        tr = Lb.rollout(m, x0, T, h); E = H_true(tr[:, 0], tr[:, 1]); dq = tr[:, 0] - ref[:, 0]
        qq, pp = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-1.5, 1.5, 31)); msk = H_true(qq, pp) < 1.8
        xg = torch.tensor(np.c_[qq[msk], pp[msk]]); fg = m(xg).detach().numpy(); ft = np.c_[pp[msk], -np.sin(qq[msk]) - GAM * pp[msk]]
        line = (f"{name} seed {seed}: train loss {loss:.2e}, field RMSE {np.sqrt(((fg-ft)**2).sum(1).mean()):.4f}, "
                f"max|q-q_ref| [0,20] {np.abs(dq[:1001]).max():.3f}, [0,60] {np.abs(dq).max():.3f}, rms [0,60] {np.sqrt((dq**2).mean()):.3f}, rms [40,60] {np.sqrt((dq[2000:]**2).mean()):.3f}; "
                f"E(20) {E[1000]:.4f} E(40) {E[2000]:.4f} E(60) {E[-1]:.4f}, max|E-E_ref| {np.abs(E-Eref).max():.4f}, max|q| on [50,60] {np.abs(tr[2500:,0]).max():.3f}, cpu {cpu:.0f} s")
        if name != "mlp":
            Hh = m.H(torch.tensor(tr)).detach().numpy(); line += f", H_th change over T {Hh[-1]-Hh[0]:+.3e}, max|H_th-H_th(0)| {np.abs(Hh-Hh[0]).max():.2e}, increases {int((np.diff(Hh[::50])>1e-9).sum())}/{len(Hh[::50])-1}"
        if name == "dhnn":
            R = m.R().detach().numpy(); line += f", R = [[{R[0,0]:.4f},{R[0,1]:.4f}],[{R[1,0]:.4f},{R[1,1]:.4f}]], eig {np.linalg.eigvalsh(R)}"
        print(line, flush=True)
        np.save(f"../../../ch09/solutions/a/s91_5_{name}_{seed}.npy", np.c_[tr, E])
