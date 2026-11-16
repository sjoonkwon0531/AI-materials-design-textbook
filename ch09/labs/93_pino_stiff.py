"""Lab 9.3-4 (supporting check): why the PINO lab uses nu = 0.02.
At nu = 0.005 (the data of 93_fno_burgers.py) the fronts are 2-3 grid points wide at s = 128, the spectral
residual is under-resolved, and instance-wise fine-tuning lowers the residual while RAISING the error."""
import os, sys, time, math, copy, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models")
c_all = time.process_time()
d = np.load(os.path.join(HERE, "..", "data", "93_burgers.npz")); traj = torch.tensor(d["traj"]); NU, T, NF = 0.005, 0.25, 512
tgrid = torch.linspace(0, T, 11)


def G(net, a, t):
    tt = (t / T).view(-1, 1, 1).expand(-1, 1, a.shape[-1])
    return a + tt * net(torch.cat([a, tt], 1))


def residual(net, a, t):
    u, ut = torch.func.jvp(lambda s: G(net, a, s), (t,), (torch.ones_like(t),))
    N = a.shape[-1]; k = 2 * math.pi * torch.arange(N // 2 + 1); uh = torch.fft.rfft(u)
    return ut + u * torch.fft.irfft(1j * k * uh, n=N) - NU * torch.fft.irfft(-k**2 * uh, n=N)


torch.manual_seed(0); net = M.FNO1d(in_ch=2, width=32, kmax=16, depth=4); g = torch.Generator().manual_seed(0)
opt = torch.optim.Adam(net.parameters(), 2e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=1500, pct_start=0.1)
for it in range(1500):                                       # data-only operator, 50 labelled trajectories, s = 64
    i = torch.randint(0, 50, (16,), generator=g); j = torch.randint(1, 11, (16,), generator=g)
    loss = M.rel_l2(G(net, traj[i, 0, None, ::8], tgrid[j]), traj[i, j, None, ::8])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
with torch.no_grad():
    e = M.rel_l2(G(net, traj[2000:, 0, None, ::8], torch.full((200,), T)), traj[2000:, -1, None, ::8]).item()
print(f"nu={NU}: data-only operator (50 labelled trajectories): test rel-L2 {e:.4f}")
for sres in (128, 256):
    for j in range(3):                                       # instance-wise fine-tuning on the residual at s = sres
        ft = copy.deepcopy(net); o = torch.optim.Adam(ft.parameters(), 2e-4); gg = torch.Generator().manual_seed(0)
        a = traj[2000 + j:2001 + j, 0, None, ::NF // sres]; a64 = traj[2000 + j:2001 + j, 0, None, ::8]; tru = traj[2000 + j, -1, ::8]
        err = lambda: ((G(ft, a64, torch.full((1,), T))[0, 0] - tru).norm() / tru.norm()).item()
        with torch.no_grad(): e0 = err()
        for it in range(200):
            t = torch.rand(16, generator=gg) * T
            l = residual(ft, a.expand(16, -1, -1), t).pow(2).mean()
            if it == 0: r0 = l.item()
            o.zero_grad(); l.backward(); o.step()
        with torch.no_grad(): e1 = err()
        print(f"  residual grid {sres}, test input {j}: residual MSE {r0:6.2f} -> {l.item():5.2f}   rel-L2 error {e0:.4f} -> {e1:.4f}")
print(f"CPU {time.process_time()-c_all:.0f} s")
