"""Ex 9.3.6 follow-up: why the flux-form model blows up in rollout (per-mode gains, growth of high-k amplitude)."""
import os, sys, math, importlib
import numpy as np, torch
torch.set_num_threads(1)
LAB = "../ch09/labs"; sys.path.insert(0, LAB); M = importlib.import_module("93_models")
d = np.load(os.path.join(LAB, "..", "data", "93_ks.npz")); traj = torch.tensor(d["traj"]); NTR, L, N = int(d["ntr"]), float(d["L"]), traj.shape[-1]; te = traj[NTR:]
make = lambda cons=None: M.FNO1d(width=32, kmax=24, depth=4, residual=True, conserve=cons, L=64.0)
sd = torch.load(os.path.join(LAB, "93_rollout_models.pt")); nets = {"cons": make("mean"), "flux": make("flux")}
nets["cons"].load_state_dict(sd["cons"]); nets["flux"].load_state_dict(torch.load("../ch09/solutions/a/s93_6_flux.pt"))
x = torch.arange(N) * L / N; EPS = 1e-2; states = te[:, 100].double()
def gain(step, ks):
    base = step(states); g = []
    for k in ks:
        r = 0.0
        for ph in (0.0, math.pi / 2):
            dlt = EPS * torch.cos(2 * math.pi * k * x / L + ph).double()
            r += (torch.fft.rfft(step(states + dlt) - base)[:, k].abs() / (EPS * N / 2)).mean().item() / 2
        g.append(r)
    return g
ks = [4, 7, 12, 20, 26, 30, 40, 50, 60, 63]
with torch.no_grad():
    for n in ("cons", "flux"):
        g = gain(lambda u, n=n: nets[n](u.float()[:, None])[:, 0].double(), ks)
        print(f"[{n}] empirical gain on attractor states: " + "  ".join(f"k={k}: {v:.3f}" for k, v in zip(ks, g)))
    # c(x): local linear response of the flux to a high-k perturbation -> |g|^2 = 1 + (q c)^2 ?
    fl = nets["flux"]; u = te[:, 100][:, None]
    for k in (30, 50, 60):
        q = 2 * math.pi * k / L; dl = EPS * torch.cos(2 * math.pi * k * x / L).float()
        dJ = fl.proj(fl.features(u + dl))[:, 0] - fl.proj(fl.features(u))[:, 0]
        c = (torch.fft.rfft(dJ)[:, k].abs() / (EPS * N / 2)).mean().item()
        print(f"   flux response |dJ_k/du_k| at k={k}: c = {c:.4f}, q c = {q*c:.3f}, sqrt(1+(qc)^2) = {math.sqrt(1+(q*c)**2):.3f}")
    # rollout: high-k amplitude and mean drift versus time, blow-up times
    for n in ("cons", "flux"):
        net = nets[n]; u = te[:, 0][:, None]; hk = {}; md = {}; tb = torch.full((16,), -1)
        for t in range(1, 2001):
            u = net(u); u = torch.nan_to_num(u, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            big = u[:, 0].abs().amax(-1) > 20; tb = torch.where((tb < 0) & big, torch.tensor(t), tb)
            if t in (1, 10, 20, 30, 40, 50, 60, 80, 100):
                ok = tb < 0
                a = (torch.fft.rfft(u[:, 0]).abs()[:, 33:] ** 2).sum(-1).sqrt() / N
                hk[t] = (a[ok].median().item() if ok.any() else float("nan"), a.max().item()); md[t] = ((u[:, 0].mean(-1) - te[:, 0].mean(-1)).abs()[ok].max().item() if ok.any() else float("nan"), int(ok.sum()))
        tbs = sorted(tb[tb > 0].tolist())
        print(f"[{n}] high-k amplitude (k>32), median over trajectories still bounded / max: " + "  ".join(f"t={t}: {v[0]:.1e}/{v[1]:.1e}" for t, v in hk.items()))
        print(f"[{n}] max mean drift over bounded trajectories (number bounded): " + "  ".join(f"t={t}: {v[0]:.1e} ({v[1]})" for t, v in md.items()))
        print(f"[{n}] blow-up times (|u|>20): {tbs if tbs else 'none'}" + (f"; median {np.median(tbs):.0f}" if tbs else ""))
    # which wavenumber carries the instability: spectrum of a flux rollout just before blow-up
    net = nets["flux"]; u = te[:1, 0][:, None]; 
    for t in range(1, 200):
        u = net(u)
        if u.abs().max() > 8: break
    sp = torch.fft.rfft(u[0, 0]).abs() / N
    print(f"[flux] trajectory 0 at t={t} (max|u|={u.abs().max():.1f}): amplitude by band k<=24: {sp[:25].norm():.2e}, 25-32: {sp[25:33].norm():.2e}, 33-48: {sp[33:49].norm():.2e}, 49-64: {sp[49:].norm():.2e}; peak k = {int(sp[1:].argmax())+1}")
