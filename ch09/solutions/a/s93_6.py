"""Ex 9.3.6: flux-form conservative FNO (conserve="flux") vs zero-mode projection (conserve="mean") on KS."""
import os, sys, time, math, importlib, copy
import numpy as np, torch
torch.set_num_threads(1)
LAB = "../../../ch09/labs"; sys.path.insert(0, LAB); M = importlib.import_module("93_models")
d = np.load(os.path.join(LAB, "..", "data", "93_ks.npz"))
traj = torch.tensor(d["traj"]); NTR, NS, L, N = int(d["ntr"]), int(d["ns_tr"]), float(d["L"]), traj.shape[-1]
X = traj[:NTR, :NS + 1]; te = traj[NTR:]
KMAX, WIDTH, BATCH, PRE_EP, FT_UPD = 24, 32, 64, 6, 600
make = lambda cons=None: M.FNO1d(width=WIDTH, kmax=KMAX, depth=4, residual=True, conserve=cons, L=64.0)
def batch(g, K):
    i = torch.randint(0, NTR, (BATCH,), generator=g); n = torch.randint(0, NS - K + 1, (BATCH,), generator=g)
    return [X[i, n + j][:, None] for j in range(K + 1)]
def run(net, updates, lr, seed):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=updates, pct_start=0.1); c0 = time.process_time()
    for it in range(updates):
        u = batch(g, 1); loss = M.rel_l2(net(u[0]), u[1]); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return time.process_time() - c0
@torch.no_grad()
def onestep_err(net): return M.rel_l2(net(te[:, :-1].reshape(-1, 1, N)), te[:, 1:].reshape(-1, 1, N)).item()
PT = "../../../ch09/solutions/a/s93_6_flux.pt"
if not os.path.exists(PT):
    pre_upd = PRE_EP * (NTR * NS // BATCH)
    torch.manual_seed(0); net = make("flux"); c1 = run(net, pre_upd, 2e-3, seed=1); e1 = onestep_err(net)
    c2 = run(net, FT_UPD, 5e-4, seed=2); print(f"flux: pre-train {pre_upd} updates: one-step test rel-L2 {e1:.4f} ({c1:.0f} s CPU); fine-tune {FT_UPD}: {onestep_err(net):.4f} ({c2:.0f} s CPU)", flush=True)
    torch.save(net.state_dict(), PT)
sd = torch.load(os.path.join(LAB, "93_rollout_models.pt"))
nets = {"onestep": make(None), "cons": make("mean"), "flux": make("flux")}
nets["onestep"].load_state_dict(sd["onestep"]); nets["cons"].load_state_dict(sd["cons"]); nets["flux"].load_state_dict(torch.load(PT))
def lowpass(u, kc):
    uh = torch.fft.rfft(u); uh[..., kc + 1:] = 0; return torch.fft.irfft(uh, n=u.shape[-1])
@torch.no_grad()
def rollout(net, u0, steps, post=None):
    out = [u0]; u = u0[:, None]
    for _ in range(steps):
        u = net(u)
        if post is not None: u = post(u)
        u = torch.nan_to_num(u, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6); out.append(u[:, 0])
    return torch.stack(out, 1)
def corr(p, t):
    p = p - p.mean(-1, keepdim=True); t = t - t.mean(-1, keepdim=True)
    return (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + 1e-12)
def valid_time(p, t):
    c = corr(p, t) < 0.9; return torch.where(c.any(1), c.float().argmax(1), torch.tensor(c.shape[1] - 1)).float()
highk = lambda u: (torch.fft.rfft(u).abs()[..., 33:] ** 2).sum(-1).sqrt() / N
for name, net, post in (("onestep", nets["onestep"], None), ("cons(mean)", nets["cons"], None), ("flux", nets["flux"], None), ("flux+filter", nets["flux"], lambda u: lowpass(u, 32))):
    p = rollout(net, te[:, 0], 2000, post); vt = valid_time(p[:, :401], te)
    err = ((p[:, :401] - te).norm(dim=-1) / te.norm(dim=-1)).mean(0); blow = (p.abs().amax(-1) > 20).any(1)
    tb = [int((p[i].abs().amax(-1) > 20).float().argmax()) for i in range(len(p)) if blow[i]]
    mass = (p.mean(-1) - te[:, :1].mean(-1)).abs().mean(0); hk = highk(p[:, 1000:]).mean(1)
    print(f"{name:12s}: one-step {onestep_err(net):.4f} err(1) {err[1]:.4f} err(10) {err[10]:.3f} | valid time {vt.mean():5.1f} +/- {vt.std():4.1f} | blow-ups in 2000 steps {int(blow.sum())}/16"
          + (f" (first at t={min(tb)})" if tb else "") + f" | mean drift t=400 {mass[400]:.1e}, t=2000 {mass[2000]:.1e} | high-k amplitude (k>32) median {hk.median():.1e} max {hk.max():.1e}", flush=True)
# von Neumann factor at u = 0
for name in ("cons", "flux"):
    J = torch.autograd.functional.jacobian(lambda u, n=name: nets[n](u[None, None])[0, 0], torch.zeros(N)); g0 = torch.fft.fft(J[:, 0]).abs()[: N // 2 + 1]
    print(f"[{name}] |g(k)| at u=0: k=0: {g0[0]:.6f}; k=25: {g0[25]:.4f}, k=40: {g0[40]:.4f}, k=55: {g0[55]:.4f}, k=63: {g0[63]:.4f}, k=64: {g0[64]:.4f}; max over k>24: {g0[25:].max():.4f}; max_k {g0.max():.3f} at k={int(g0.argmax())}")
# local conservation: sum of the increment over a sub-interval equals the flux difference (flux model only)
with torch.no_grad():
    fl = nets["flux"]; u = te[:, 100][:, None]; Jt = fl.proj(fl.features(u))[:, 0]           # learned (time-integrated) flux on the grid
    inc = fl(u)[:, 0] - u[:, 0]
    k = 2 * math.pi / L * torch.arange(N // 2 + 1); iq = 1j * k
    def clean(J):                                                                             # remove the mean and the Nyquist mode
        Jh = torch.fft.rfft(J); Jh[..., 0] = 0; Jh[..., -1] = 0; return torch.fft.irfft(Jh, n=N)
    # (d) compare with the exact time-integrated flux of the solver and with the instantaneous formula
    un, up = te[:, :-1].reshape(-1, N), te[:, 1:].reshape(-1, N)
    dh = torch.fft.rfft(up - un); Jbar_h = torch.zeros_like(dh); Jbar_h[:, 1:-1] = -dh[:, 1:-1] / iq[1:-1]; Jbar = torch.fft.irfft(Jbar_h, n=N)
    def Jinst(u):
        uh = torch.fft.rfft(u); return 0.5 * u * u + torch.fft.irfft(iq * uh, n=N) + torch.fft.irfft(iq**3 * uh, n=N)
    Jth = clean(fl.proj(fl.features(un[:, None]))[:, 0]); rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean().item()
    raw = fl.proj(fl.features(un[:, None]))[:, 0]
    print(f"(d) learned flux vs exact time-integrated flux Jbar: rel L2 {rel(Jth, Jbar):.4f}; instantaneous J(u_n) vs Jbar: {rel(clean(Jinst(un)), Jbar):.3f}; J(u_n+1) vs Jbar: {rel(clean(Jinst(up)), Jbar):.3f}; trapezoid: {rel(clean(0.5*(Jinst(un)+Jinst(up))), Jbar):.3f}")
    print(f"    learned flux vs instantaneous J(u_n): {rel(Jth, clean(Jinst(un))):.3f}; without removing the mean: {rel(raw, Jinst(un)):.3f}; mean of raw J_theta {raw.mean().item():+.3f} (std over states {raw.mean(-1).std().item():.3f}), mean of J(u_n) {Jinst(un).mean().item():+.3f}; Nyquist amplitude of raw J_theta {(torch.fft.rfft(raw)[:, -1].abs().mean()/N).item():.2e}")
    print(f"    rms Jbar {Jbar.pow(2).mean().sqrt().item():.3f}, rms J_inst {clean(Jinst(un)).pow(2).mean().sqrt().item():.3f}")
