"""Lab 9.3-3 (evaluation): autoregressive rollouts of the KS surrogates trained by 93_rollout_train.py.
(1) rollout error / correlation time over 400 steps (16 test trajectories)
(2) long rollouts (2000 steps): blow-ups and time-averaged spectrum
(3) per-mode amplification: learned map vs solver (empirical, on attractor states; exact, at u = 0)
(4) mean (mass) drift with / without the zero-mode constraint; spectral filter at inference
"""
import os, sys, time, json, math, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models"); D = importlib.import_module("93_data")
t_all = time.time(); c_all = time.process_time()

d = np.load(os.path.join(HERE, "..", "data", "93_ks.npz"))
traj = torch.tensor(d["traj"]); NTR, L, N = int(d["ntr"]), float(d["L"]), traj.shape[-1]
te = traj[NTR:]                                                               # (16, 401, 128)
sd = torch.load(os.path.join(HERE, "93_rollout_models.pt"))
KMAX, CORR_TH = 24, 0.9
names = ["onestep", "noise", "push", "unroll", "cons"]
nets = {}
for n in names:
    nets[n] = M.FNO1d(width=32, kmax=KMAX, depth=4, residual=True, conserve="mean" if n == "cons" else None, L=L)
    nets[n].load_state_dict(sd[n]); nets[n].eval()


def lowpass(u, kc):
    uh = torch.fft.rfft(u); uh[..., kc + 1:] = 0
    return torch.fft.irfft(uh, n=u.shape[-1])


@torch.no_grad()
def rollout(net, u0, steps, post=None):
    out = [u0]; u = u0[:, None]
    for _ in range(steps):
        u = net(u)
        if post is not None: u = post(u)
        u = torch.nan_to_num(u, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        out.append(u[:, 0])
    return torch.stack(out, 1)


def corr(p, t):
    p = p - p.mean(-1, keepdim=True); t = t - t.mean(-1, keepdim=True)
    return (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + 1e-12)


def valid_time(p, t):
    c = corr(p, t) < CORR_TH                                                  # (B, steps+1)
    return torch.where(c.any(1), c.float().argmax(1), torch.tensor(c.shape[1] - 1)).float()


highk = lambda u: (torch.fft.rfft(u).abs()[..., 33:] ** 2).sum(-1).sqrt() / N   # amplitude above k = 32
out = {"names": names}
spec = lambda u: torch.fft.rfft(u).abs().pow(2).mean(tuple(range(u.dim() - 1))).sqrt() / N
out["spec_true"] = spec(te).tolist()
print(f"KS test set: {te.shape[0]} trajectories x {te.shape[1]-1} steps of dt=1; correlation threshold {CORR_TH}")

# ---------------- (1), (2) rollouts ----------------
variants = [(n, nets[n], None) for n in names]
variants.append(("onestep+P0", nets["onestep"], lambda u: u - u.mean(-1, keepdim=True) + te[:, 0].mean(-1)[:, None, None]))
variants.append(("onestep+filter", nets["onestep"], lambda u: lowpass(u, 32)))
for name, net, post in variants:
    p = rollout(net, te[:, 0], 2000, post)
    vt = valid_time(p[:, :401], te)
    err = ((p[:, :401] - te).norm(dim=-1) / te.norm(dim=-1)).mean(0)
    blow = (p.abs().amax(-1) > 20).any(1)
    tb = [(int((p[i].abs().amax(-1) > 20).float().argmax())) for i in range(len(p)) if blow[i]]
    ok = ~blow
    sp = spec(p[ok][:, 1000:]) if ok.any() else torch.full((N // 2 + 1,), float("nan"))
    mass = (p[:, :401].mean(-1) - te[:, :1].mean(-1)).abs().mean(0)
    hk = highk(p[:, 1000:]).mean(1)                                           # per-trajectory high-k amplitude, t = 1000..2000
    spt = torch.fft.rfft(p[:, 1000:]).abs().pow(2).mean(1).sqrt() / N              # per-trajectory rms spectrum (16, 65)
    out[name] = dict(valid=vt.tolist(), err=err.tolist(), blow=int(blow.sum()), t_blow=tb, spec=sp.tolist(), mass=mass.tolist(),
                     spec_med=spt.median(0).values.tolist(), spec_worst=spt[hk.argmax()].tolist(),
                     corr=corr(p[:, :401], te).mean(0).tolist(), hk_med=hk.median().item(), hk_max=hk.max().item())
    print(f"{name:15s}: err(1) {err[1]:.4f} err(10) {err[10]:.3f} | valid time {vt.mean():5.1f} +/- {vt.std():4.1f} | blow-ups in 2000 steps {int(blow.sum())}/16"
          + (f" (first at t={min(tb)})" if tb else "") + f" | mean drift(t=400) {mass[400]:.1e} | high-k amplitude (k>32) median {hk.median():.1e} max {hk.max():.1e}")
out["hk_true"] = highk(te).mean(1).median().item()
out["spec_true_med"] = (torch.fft.rfft(te).abs().pow(2).mean(1).sqrt() / N).median(0).values.tolist()
print(f"solver         : high-k amplitude (k>32) median over test trajectories {out['hk_true']:.1e}")

# twin experiment: the EXACT solver continued from the FNO's one-step prediction (same initial error, perfect model afterwards)
with torch.no_grad(): u1 = nets["onestep"](te[:, 0, None])[:, 0]
tw = torch.tensor(D.ks_solve(u1.numpy().astype(float), L, 200.0, 0.025, save_every=40))[:, 0:201]   # t = 1..201
ref = te[:, 1:202].double()
cc = corr(tw, ref) < CORR_TH
vt_tw = torch.where(cc.any(1), cc.float().argmax(1), torch.tensor(200)).float() + 1
e_tw = ((tw - ref).norm(dim=-1) / ref.norm(dim=-1))
lyap = (torch.log(e_tw[:, 40] / e_tw[:, 10]).mean() / 30).item()
out.update(valid_twin=vt_tw.tolist(), lyap=lyap, err_twin=e_tw.mean(0).tolist())
print(f"twin (solver restarted from the FNO one-step prediction): valid time {vt_tw.mean():.1f} +/- {vt_tw.std():.1f}; "
      f"error growth rate between t=11 and t=41: {lyap:.3f} per unit time (e-folding time {1/lyap:.1f})")
out["example_true"] = te[0, :201].tolist()
out["example_onestep"] = rollout(nets["onestep"], te[:1, 0], 200)[0].tolist()
out["example_unroll"] = rollout(nets["unroll"], te[:1, 0], 200)[0].tolist()

# ---------------- error spectrum versus time ----------------
with torch.no_grad():
    for name in ("onestep", "unroll"):
        p = rollout(nets[name], te[:, 0], 40)
        out[f"errspec_{name}"] = {str(n): spec(p[:, n] - te[:, n]).tolist() for n in (1, 5, 20, 40)}

# ---------------- (3) amplification factors ----------------
x = torch.arange(N) * L / N; EPS = 1e-2
states = te[:, 100].double()                                                  # 16 attractor states
ks = list(range(1, N // 2))
def gain(step):
    g = []
    base = step(states)
    for k in ks:
        r = 0.0
        for ph in (0.0, math.pi / 2):
            dlt = EPS * torch.cos(2 * math.pi * k * x / L + ph).double()
            dh = torch.fft.rfft(step(states + dlt) - base)[:, k].abs() / (EPS * N / 2)
            r = r + dh.mean().item() / 2
        g.append(r)
    return g
solver_step = lambda u: torch.tensor(D.ks_solve(u.numpy(), L, 1.0, 0.025))
out["k"] = ks; out["gain_solver"] = gain(solver_step)
with torch.no_grad():
    for name in names:
        out[f"gain_{name}"] = gain(lambda u, n=name: nets[n](u.float()[:, None])[:, 0].double())
q = 2 * math.pi * np.array(ks) / L
out["gain_linear_exact"] = np.exp(q**2 - q**4).tolist()
# exact von Neumann factor of the learned map at the constant state u = 0 (Jacobian is circulant)
for name in ("onestep", "unroll"):
    f = lambda u, n=name: nets[n](u[None, None])[0, 0]
    J = torch.autograd.functional.jacobian(f, torch.zeros(N))                 # (N, N)
    circ = max((torch.roll(J[:, j], -j) - J[:, 0]).abs().max().item() for j in range(N))
    g0 = torch.fft.fft(J[:, 0]).abs()[: N // 2 + 1]
    out[f"g0_{name}"] = g0.tolist(); out[f"circ_{name}"] = circ
    hi = g0[KMAX + 1:]
    print(f"[{name}] Jacobian at u=0: circulant defect {circ:.1e}; |g(k)| for k>kmax: min {hi.min():.4f} max {hi.max():.4f} (flat); "
          f"max_k |g| = {g0.max():.3f} at k={int(g0.argmax())} (exact linear KS: {np.exp(0.25):.3f} at q=0.707, k=7.2)")
for kk in (4, 7, 12, 20, 30, 50):
    print(f"  k={kk:2d} (q={2*math.pi*kk/L:.2f}): empirical gain  solver {out['gain_solver'][kk-1]:.2e} | "
          + " | ".join(f"{n} {out['gain_'+n][kk-1]:.2e}" for n in ("onestep", "noise", "push", "unroll")))

# ---------------- cost ----------------
with torch.no_grad():
    t0 = time.process_time(); rollout(nets["onestep"], te[:, 0], 100); t_net = (time.process_time() - t0) / 100 / 16
t0 = time.process_time(); D.ks_solve(te[:, 0].numpy().astype(float), L, 100.0, 0.025); t_sol = (time.process_time() - t0) / 100 / 16
t0 = time.process_time(); c4 = D.ks_solve(te[:, 0].numpy().astype(float), L, 100.0, 0.1); t_sol4 = (time.process_time() - t0) / 100 / 16
e4 = float(np.mean(np.linalg.norm(D.ks_solve(te[:, 0].numpy().astype(float), L, 1.0, 0.1) - te[:, 1].numpy(), axis=1) / np.linalg.norm(te[:, 1].numpy(), axis=1)))
out.update(t_net=t_net, t_sol=t_sol, t_sol4=t_sol4, e_sol4=e4)
print(f"CPU cost per trajectory per unit time: FNO step {t_net*1e3:.3f} ms | solver, 40 steps of 0.025: {t_sol*1e3:.3f} ms | "
      f"solver, 10 steps of 0.1: {t_sol4*1e3:.3f} ms (one-step rel-L2 {e4:.1e})")
json.dump(out, open(os.path.join(HERE, "93_rollout_eval.json"), "w"))
print(f"wall {time.time()-t_all:.0f} s (CPU {time.process_time()-c_all:.0f} s)")
