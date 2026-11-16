"""Exercise 9.6.4.
  a : retrain the zero-mode FNO (cons) WITHOUT the pairs whose input is c(0) (3,040 pairs, same 5 epochs)
  b : quench-only models  c(0) -> c(2.5), 160 pairs: raw input vs normalised fluctuation (c - m)/std plus an m channel
  eval : print everything
Usage: python ex964_noquench.py a | b | eval  [smoke]"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import ch_common as cc
fno = cc.fno
MODE = sys.argv[1]; SMOKE = len(sys.argv) > 2
BATCH, LR, MODES, WIDTH = 8, 2e-3, 16, 20


class QuenchFNO(fno.FNO2d):
    """Input channels: normalised fluctuation z = (c - m)/std(c - m) and the mean m (constant field).
    Output: c + (increment - its mean), so the mass is still conserved exactly."""
    def __init__(self, **kw):
        super().__init__(in_channels=2, **kw)
    def forward(self, c):
        B, N1, N2 = c.shape
        m = c.mean(dim=(-2, -1), keepdim=True); z = (c - m) / (c - m).std(dim=(-2, -1), keepdim=True)
        g1, g2 = self.grid(B, N1, N2, c.device, c.dtype)
        v = self.lift(torch.cat([z.unsqueeze(-1), m.expand(B, N1, N2).unsqueeze(-1), g1, g2], dim=-1))
        for l in range(self.n_layers):
            v = self.spec[l](v) + self.lin[l](v)
            if l < self.n_layers - 1:
                v = F.gelu(v)
        d = self.proj2(F.gelu(self.proj1(v))).squeeze(-1)
        return c + d - d.mean(dim=(-2, -1), keepdim=True)


def fit(model, x, y, epochs, tag):
    steps = epochs * (len(x) // BATCH)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    c0 = time.process_time()
    for ep in range(epochs):
        perm = torch.randperm(len(x)); run = 0.0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = fno.rel_l2_loss(model(x[idx]), y[idx]); loss.backward(); opt.step(); sched.step(); run += loss.item()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[{tag}] epoch {ep + 1}: train loss {run / (len(perm) // BATCH):.4f} | CPU {time.process_time() - c0:.0f} s", flush=True)
    return time.process_time() - c0, steps

sfx = "_smoke" if SMOKE else ""
if MODE == "a":
    torch.manual_seed(0); np.random.seed(0)
    tr = cc.train[:8] if SMOKE else cc.train
    x = torch.from_numpy(tr[:, 1:-1].reshape(-1, 64, 64)); y = torch.from_numpy(tr[:, 2:].reshape(-1, 64, 64))
    model = fno.FNO2d(modes=MODES, width=WIDTH, n_layers=4, proj=64, conserve=True)
    print(f"[noquench] {len(x)} training pairs (inputs c(2.5) ... c(47.5))")
    cpu, steps = fit(model, x, y, 1 if SMOKE else 5, "noquench")
    torch.save({"state_dict": model.state_dict(), "cpu": cpu, "steps": steps}, cc.OUT + f"ex964_noquench{sfx}.pt")
elif MODE == "b":
    tr = cc.train[:16] if SMOKE else cc.train
    x = torch.from_numpy(tr[:, 0]); y = torch.from_numpy(tr[:, 1])
    for tag in ("raw", "norm"):
        torch.manual_seed(0); np.random.seed(0)
        model = fno.FNO2d(modes=MODES, width=WIDTH, n_layers=4, proj=64, conserve=True) if tag == "raw" else QuenchFNO(modes=MODES, width=WIDTH, n_layers=4, proj=64)
        cpu, steps = fit(model, x, y, 2 if SMOKE else 15, "quench-" + tag)
        torch.save({"state_dict": model.state_dict(), "cpu": cpu, "steps": steps}, cc.OUT + f"ex964_quench_{tag}{sfx}.pt")
else:
    test = cc.test
    print("=== (a) ===")
    cc.evaluate(cc.lab_model("cons"), "cons (Lab 9.20): all 3,200 pairs, 2,000 steps")
    ck = torch.load(cc.OUT + f"ex964_noquench{sfx}.pt")
    m = fno.FNO2d(modes=MODES, width=WIDTH, n_layers=4, proj=64, conserve=True); m.load_state_dict(ck["state_dict"]); m.eval()
    cc.evaluate(fno.to_double(m), f"cons without the quench pairs: 3,040 pairs, {ck['steps']} steps, training CPU {ck['cpu']:.0f} s")
    print("=== (b) quench step c(0) -> c(2.5), 40 test trajectories ===")
    c0, c1 = test[:, 0], test[:, 1]
    stats = lambda c: (np.std(c - c.mean((1, 2), keepdims=True), axis=(1, 2)).mean(), cc.sol.length(c).mean(), cc.sol.free_energy(c).mean(), np.abs(c).max())
    print("  model                                  first-step rel. L2   std(c - m)   L(2.5)   F(2.5)   max|c|")
    print("  solver                                 0                    %.3f        %.3f    %.3f    %.3f" % stats(c1))
    print("  persistence c(0)                       %.4f" % cc.rel(c0, c1).mean())
    for name, tag in (("Lab free", "free"), ("Lab cons", "cons"), ("Lab cons_pf", "cons_pf")):
        p = cc.apply(cc.lab_model(tag), c0)
        print(f"  {name:38s} {cc.rel(p, c1).mean():.4f}               %.3f        %.3f    %.3f    %.3f" % stats(p))
    for name, tag in (("quench-only, raw input", "raw"), ("quench-only, (c - m)/std and m", "norm")):
        ck = torch.load(cc.OUT + f"ex964_quench_{tag}{sfx}.pt")
        m = fno.FNO2d(modes=MODES, width=WIDTH, n_layers=4, proj=64, conserve=True) if tag == "raw" else QuenchFNO(modes=MODES, width=WIDTH, n_layers=4, proj=64)
        m.load_state_dict(ck["state_dict"]); m.eval(); m = fno.to_double(m)
        p = cc.apply(m, c0); e = cc.rel(p, c1)
        print(f"  {name:38s} {e.mean():.4f} (+- {e.std():.3f})    %.3f        %.3f    %.3f    %.3f   [{ck['steps']} steps, CPU {ck['cpu']:.0f} s]" % stats(p))
        ptr = cc.apply(m, cc.train[:40, 0].astype(float)); print(f"      error on 40 training quenches: {cc.rel(ptr, cc.train[:40, 1].astype(float)).mean():.4f}")
        if tag == "norm":      # hand over to the Lab pushforward model: rollout from the predicted c(2.5)
            r = cc.rollout(cc.lab_model("cons_pf"), p, cc.NREC - 1); Fq, Lq, _ = cc.diagnostics(r)
            print(f"      then cons_pf from the predicted c(2.5): L(50) = {Lq[:, -1].mean():.3f} (solver {cc.L_s[:, -1].mean():.3f}), F(50) = {Fq[:, -1].mean():.3f}, max|c| = {np.abs(r).max():.2f},"
                  f" slope {cc.slope(cc.t1, Lq.mean(0)):.3f}")
