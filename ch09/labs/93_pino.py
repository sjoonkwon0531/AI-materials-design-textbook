"""Lab 9.3-4: PINO-style training and instance-wise fine-tuning with a time-conditioned FNO-1D.
Burgers, nu = 0.02, T = 0.25:  G_theta(u0, t)(x) = u0(x) + (t/T) N_theta(u0, t/T)(x)   (hard initial condition)
(1) operator learning with few labelled trajectories: data loss only  vs  data + PDE residual (unlabelled inputs, finer grid)
(2) instance-wise fine-tuning on the PDE residual, starting from each operator and from a random initialisation
The residual uses spectral x-derivatives and forward-mode AD in t (torch.func.jvp).
"""
import os, sys, time, json, math, copy, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models"); D = importlib.import_module("93_data")
t_all = time.time(); c_all = time.process_time()
NU, T, NF, S_DATA, S_PDE = 0.02, 0.25, 256, 64, 128

# ---------------- data: 100 labelled + 200 test trajectories, 4000 unlabelled initial conditions ----------------
rng = np.random.default_rng(95)
a0 = D.grf(300, NF, rng); t0 = time.process_time()
traj = torch.tensor(D.burgers_solve(a0, NU, T, 1e-3, save_every=25), dtype=torch.float32)   # (300, 11, 256)
t_solver = (time.process_time() - t0) / 300                   # CPU time per trajectory
A_unl = torch.tensor(D.grf(4000, NF, rng), dtype=torch.float32)
tgrid = torch.linspace(0, T, 11); TE = slice(100, 300)


def G(net, a, t):                                            # a: (B, 1, N), t: (B,)
    tt = (t / T).view(-1, 1, 1).expand(-1, 1, a.shape[-1])
    return a + tt * net(torch.cat([a, tt], 1))               # hard initial condition


def residual(fn, a, t):                                      # r = u_t + u u_x - nu u_xx on the grid of a
    u, ut = torch.func.jvp(lambda s: fn(a, s), (t,), (torch.ones_like(t),))      # forward-mode d/dt
    N = a.shape[-1]; k = 2 * math.pi * torch.arange(N // 2 + 1); uh = torch.fft.rfft(u)
    ux, uxx = torch.fft.irfft(1j * k * uh, n=N), torch.fft.irfft(-k**2 * uh, n=N)
    return ut + u * ux - NU * uxx


@torch.no_grad()
def test(net):
    a = traj[TE, 0, None, ::NF // S_DATA]
    return M.rel_l2_each(G(net, a, torch.full((200,), T)), traj[TE, -1, None, ::NF // S_DATA])


def res_mse(net):
    a = traj[TE, 0, None, ::NF // S_PDE][:64]; t = torch.linspace(0.02, 1, 64) * T
    return residual(lambda a_, s: G(net, a_, s), a, t).pow(2).mean().item()


# sanity check of the residual routine on the exact Cole-Hopf solution (float64)
x = (torch.arange(128, dtype=torch.float64) / 128).expand(4, 1, -1)
def exact(a_, s):
    e = torch.exp(-4 * math.pi**2 * NU * s).view(-1, 1, 1)
    return 4 * math.pi * NU * e * torch.sin(2 * math.pi * x) / (1.5 + e * torch.cos(2 * math.pi * x))
r = residual(exact, x, torch.tensor([0.05, 0.1, 0.2, 0.25], dtype=torch.float64))
print(f"residual routine on the exact Cole-Hopf solution: max|r| = {r.abs().max():.1e}")


def train(mode, nlab, updates=1500, lam=0.03, seed=0):
    torch.manual_seed(seed); net = M.FNO1d(in_ch=2, width=32, kmax=16, depth=4)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=updates, pct_start=0.1)
    c0 = time.process_time()
    for it in range(updates):
        loss = 0.0
        if mode in ("data", "both"):                         # labelled snapshots at s = 64
            i = torch.randint(0, nlab, (16,), generator=g); j = torch.randint(1, 11, (16,), generator=g)
            loss = loss + M.rel_l2(G(net, traj[i, 0, None, ::NF // S_DATA], tgrid[j]), traj[i, j, None, ::NF // S_DATA])
        if mode in ("pde", "both"):                          # residual on unlabelled inputs at s = 128, random times
            i = torch.randint(0, 4000, (16,), generator=g); t = torch.rand(16, generator=g) * T
            loss = loss + lam * residual(lambda a_, s: G(net, a_, s), A_unl[i, None, ::NF // S_PDE], t).pow(2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return net, time.process_time() - c0


def finetune(net0, j, steps=200, lr=3e-4, nt=8):
    """instance-wise: minimise the residual of ONE test input on a fixed set of nt times."""
    net = copy.deepcopy(net0); opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    a = traj[100 + j:101 + j, 0, None, ::NF // S_PDE].expand(nt, -1, -1); t = (torch.arange(nt) + 0.5) / nt * T
    a64 = traj[100 + j:101 + j, 0, None, ::NF // S_DATA]; tru = traj[100 + j, -1, ::NF // S_DATA]
    err = lambda: ((G(net, a64, torch.full((1,), T))[0, 0] - tru).norm() / tru.norm()).item()
    hist = []
    for it in range(steps):
        loss = residual(lambda a_, s: G(net, a_, s), a, t).pow(2).mean()
        if it % 20 == 0:
            with torch.no_grad(): hist.append((it, err(), loss.item()))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad(): hist.append((steps, err(), residual(lambda a_, s: G(net, a_, s), a, t).pow(2).mean().item()))
    return hist


out = {"t_solver": t_solver}
print(f"Burgers nu={NU}, T={T}: solver N={NF}, dt=1e-3 ({t_solver*1e3:.2f} ms CPU per trajectory); data loss at s={S_DATA}, residual at s={S_PDE}")
nets = {}
for mode, nlab in (("data", 25), ("both", 25), ("data", 100), ("both", 100), ("pde", 0)):
    net, cpu = train(mode, nlab); e = test(net); key = f"{mode}{nlab}"; nets[key] = net
    out[key] = dict(err=e.mean().item(), err_med=e.median().item(), res=res_mse(net), cpu=cpu)
    print(f"operator [{mode:4s}, {nlab:3d} labelled]: test rel-L2 {e.mean():.4f} (median {e.median():.4f})  residual MSE {out[key]['res']:8.3f}   ({cpu:.0f} s CPU)")

torch.manual_seed(1); nets["random"] = M.FNO1d(in_ch=2, width=32, kmax=16, depth=4)
NI = 6
for key in ("data25", "both25", "pde0", "random"):
    c0 = time.process_time(); H = [finetune(nets[key], j, lr=3e-4 if key != "random" else 2e-3) for j in range(NI)]
    cpu = (time.process_time() - c0) / NI
    e0 = np.array([h[0][1] for h in H]); e1 = np.array([h[-1][1] for h in H]); r0 = np.array([h[0][2] for h in H]); r1 = np.array([h[-1][2] for h in H])
    out["ft_" + key] = dict(hist=H, cpu=cpu)
    print(f"fine-tune from [{key:6s}] ({NI} instances, 200 steps, {cpu:.1f} s CPU each): rel-L2 {e0.mean():.4f} -> {e1.mean():.4f}"
          f"  (per instance: " + " ".join(f"{a:.3f}>{b:.3f}" for a, b in zip(e0, e1)) + f")  residual MSE {r0.mean():.2f} -> {r1.mean():.3f}")
with torch.no_grad():
    j = 0; a = traj[100 + j:101 + j, 0, None, ::NF // S_DATA]; tt = torch.full((1,), T)
    out["example"] = dict(a=a[0, 0].tolist(), u=traj[100 + j, -1, ::NF // S_DATA].tolist(),
                          data25=G(nets["data25"], a, tt)[0, 0].tolist(), both25=G(nets["both25"], a, tt)[0, 0].tolist())
json.dump(out, open(os.path.join(HERE, "93_pino.json"), "w"))
print(f"wall {time.time()-t_all:.0f} s (CPU {time.process_time()-c_all:.0f} s)")
