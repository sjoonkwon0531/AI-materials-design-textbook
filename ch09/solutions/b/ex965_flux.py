"""Exercise 9.6.5: flux-form FNO.  The projection outputs (Jx, Jy) and the increment is d = -div J (spectral, integer wavenumbers).
Usage: python ex965_flux.py train | eval [smoke]"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import ch_common as cc
fno = cc.fno
MODE = sys.argv[1]; SMOKE = len(sys.argv) > 2
EPOCHS, BATCH, LR, MODES, WIDTH = 5, 8, 2e-3, 16, 20


class FluxFNO(fno.FNO2d):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.proj2 = nn.Linear(self.proj1.out_features, 2)             # (Jx, Jy)
    def flux(self, u):
        B, N1, N2 = u.shape
        g1, g2 = self.grid(B, N1, N2, u.device, u.dtype)
        v = self.lift(torch.cat([u.unsqueeze(-1), g1, g2], dim=-1))
        for l in range(self.n_layers):
            v = self.spec[l](v) + self.lin[l](v)
            if l < self.n_layers - 1:
                v = F.gelu(v)
        return self.proj2(F.gelu(self.proj1(v)))                       # (B, N1, N2, 2)
    def increment(self, u):
        B, N1, N2 = u.shape
        Jh = torch.fft.rfft2(self.flux(u), dim=(1, 2))                  # (B, N1, N2//2+1, 2)
        kx = torch.fft.fftfreq(N1, d=1.0 / N1).to(u.dtype); ky = torch.fft.rfftfreq(N2, d=1.0 / N2).to(u.dtype)
        kx = torch.where(kx.abs() == N1 // 2, torch.zeros_like(kx), kx)    # the derivative of the Nyquist mode is set to zero
        ky = torch.where(ky == N2 // 2, torch.zeros_like(ky), ky)
        dh = -(1j * kx.view(1, N1, 1) * Jh[..., 0] + 1j * ky.view(1, 1, -1) * Jh[..., 1])
        return torch.fft.irfft2(dh, s=(N1, N2), dim=(1, 2))             # d = -div J; its k = 0 mode is zero identically

sfx = "_smoke" if SMOKE else ""
if MODE == "train":
    torch.manual_seed(0); np.random.seed(0)
    tr = cc.train[:8] if SMOKE else cc.train
    x = torch.from_numpy(tr[:, :-1].reshape(-1, 64, 64)); y = torch.from_numpy(tr[:, 1:].reshape(-1, 64, 64))
    model = FluxFNO(modes=MODES, width=WIDTH, n_layers=4, proj=64)
    print(f"[flux] {fno.n_params(model):,d} real parameters, {len(x)} pairs")
    steps = (1 if SMOKE else EPOCHS) * (len(x) // BATCH)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    c0 = time.process_time()
    for ep in range(1 if SMOKE else EPOCHS):
        perm = torch.randperm(len(x)); run = 0.0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = fno.rel_l2_loss(model(x[idx]), y[idx]); loss.backward(); opt.step(); sched.step(); run += loss.item()
        print(f"[flux] epoch {ep + 1}: train loss {run / (len(perm) // BATCH):.4f} | CPU {time.process_time() - c0:.0f} s", flush=True)
    torch.save({"state_dict": model.state_dict(), "cpu": time.process_time() - c0, "steps": steps}, cc.OUT + f"ex965_flux{sfx}.pt")
else:
    cons = cc.lab_model("cons")
    ck = torch.load(cc.OUT + f"ex965_flux{sfx}.pt")
    m = FluxFNO(modes=MODES, width=WIDTH, n_layers=4, proj=64); m.load_state_dict(ck["state_dict"]); m.eval(); m = fno.to_double(m)
    cc.evaluate(m, f"flux form d = -div J: {ck['steps']} steps, training CPU {ck['cpu']:.0f} s")
    x = cc.test[:, 1:-1:3].reshape(-1, 64, 64)
    with torch.no_grad():
        xt = torch.from_numpy(x[:40]); d_flux = m.increment(xt).numpy(); d_cons = cons.increment(xt).numpy(); J = m.flux(xt).numpy()
    d_true = cc.test[:, 2::3].reshape(-1, 64, 64)[:40] - x[:40]
    print(f"  (a) mean of the increment: flux {np.abs(d_flux.mean((1, 2))).max():.1e}, cons {np.abs(d_cons.mean((1, 2))).max():.1e}, solver {np.abs(d_true.mean((1, 2))).max():.1e}")
    print(f"  rms J = {np.sqrt((J**2).mean()):.4f}, rms d = {np.sqrt((d_flux**2).mean()):.4f} (true increment {np.sqrt((d_true**2).mean()):.4f})")
    # (c) the Helmholtz construction: any zero-mean d equals -div J with J = -grad(lap^-1 d)... check on the cons increment
    k = np.fft.fftfreq(64, d=1 / 64); KX, KY = np.meshgrid(k, k, indexing="ij"); K2 = KX**2 + KY**2; K2[0, 0] = 1
    dh = np.fft.fft2(d_cons); Jx = np.real(np.fft.ifft2(1j * KX * dh / K2)); Jy = np.real(np.fft.ifft2(1j * KY * dh / K2))
    KXd, KYd = KX.copy(), KY.copy(); KXd[np.abs(KX) == 32] = 0; KYd[np.abs(KY) == 32] = 0
    rec = np.real(np.fft.ifft2(-(1j * KXd * np.fft.fft2(Jx) + 1j * KYd * np.fft.fft2(Jy))))
    print(f"  (c) cons increment rebuilt as -div J with J = grad(lap^-1 d): rel. difference {np.linalg.norm(rec - d_cons) / np.linalg.norm(d_cons):.1e};"
          f" rms |J| = {np.sqrt((Jx**2 + Jy**2).mean()):.4f}")
    nyq = (np.abs(KX) == 32) | (np.abs(KY) == 32)
    print(f"      share of the cons increment norm^2 in modes with |kx| = 32 or |ky| = 32: {(np.abs(dh[:, nyq])**2).sum() / (np.abs(dh)**2).sum():.1e}")
    # locality of the two increments: response to a localised perturbation of the input
    xs = x[:8].copy(); bump = np.zeros((64, 64)); bump[30:34, 30:34] = 0.2
    X, Y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij"); dist = np.hypot(np.minimum(abs(X - 31.5), 64 - abs(X - 31.5)), np.minimum(abs(Y - 31.5), 64 - abs(Y - 31.5)))
    for name, mod in (("solver", None), ("cons", cons), ("flux", m)):
        g = (lambda a: cc.sol.advance(a, 50)) if mod is None else (lambda a: cc.apply(mod, a))
        resp = np.abs(g(xs + bump) - g(xs)).mean(0)
        print(f"      response to a 4x4 bump of 0.2: {name:6s} mean |dG| within 8 cells {resp[dist < 8].mean():.2e}, beyond 24 cells {resp[dist > 24].mean():.2e}  (ratio {resp[dist > 24].mean() / resp[dist < 8].mean():.1e})")
