"""Ex 9.3.6 follow-up: flux-form model with a low-pass filter at kc = kmax = 24 (all modes with R(k) = 0 removed every step)."""
import os, sys, math, importlib
import numpy as np, torch
torch.set_num_threads(1)
LAB = "../ch09/labs"; sys.path.insert(0, LAB); M = importlib.import_module("93_models")
d = np.load(os.path.join(LAB, "..", "data", "93_ks.npz")); traj = torch.tensor(d["traj"]); NTR, L, N = int(d["ntr"]), float(d["L"]), traj.shape[-1]; te = traj[NTR:]
net = M.FNO1d(width=32, kmax=24, depth=4, residual=True, conserve="flux", L=64.0); net.load_state_dict(torch.load("../ch09/solutions/a/s93_6_flux.pt"))
def lowpass(u, kc):
    uh = torch.fft.rfft(u); uh[..., kc + 1:] = 0; return torch.fft.irfft(uh, n=u.shape[-1])
def corr(p, t):
    p = p - p.mean(-1, keepdim=True); t = t - t.mean(-1, keepdim=True); return (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + 1e-12)
with torch.no_grad():
    for kc in (24, 20):
        u = te[:, 0][:, None]; out = [te[:, 0]]
        for t in range(2000):
            u = lowpass(net(u), kc); u = torch.nan_to_num(u, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6); out.append(u[:, 0])
        p = torch.stack(out, 1); c = corr(p[:, :401], te) < 0.9; vt = torch.where(c.any(1), c.float().argmax(1), torch.tensor(400)).float()
        blow = (p.abs().amax(-1) > 20).any(1); tb = [int((p[i].abs().amax(-1) > 20).float().argmax()) for i in range(16) if blow[i]]
        mass = (p.mean(-1) - te[:, :1].mean(-1)).abs().mean(0)
        e1 = ((p[:, 1] - te[:, 1]).norm(dim=-1) / te[:, 1].norm(dim=-1)).mean().item()
        rms = p[:, 1000:].pow(2).mean().sqrt().item()
        print(f"flux + filter at k>{kc}: err(1) {e1:.4f} | valid time {vt.mean():.1f} +/- {vt.std():.1f} | blow-ups {int(blow.sum())}/16" + (f" (first at t={min(tb)})" if tb else "") + f" | mean drift t=400 {mass[400]:.1e}, t=2000 {mass[2000]:.1e} | rms u over t=1000..2000: {rms:.3f} (solver: {te.pow(2).mean().sqrt().item():.3f})")
    tef = lowpass(te, 24); print(f"content of the solver solution above k=24: {((te - tef).norm(dim=-1) / te.norm(dim=-1)).mean().item():.2e}; above 20: {((te - lowpass(te, 20)).norm(dim=-1) / te.norm(dim=-1)).mean().item():.2e}")
