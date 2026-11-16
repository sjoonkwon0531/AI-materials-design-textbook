"""Exercise 9.5.5: replace the coordinate channels (x, y) of the FNO-2D of Lab 9.17 by
   'f'    : the forcing field f(x, y) / 0.1 (one channel), or
   'trig' : sin 2 pi x, cos 2 pi x, sin 2 pi y, cos 2 pi y (four channels),
train with the same seed and budget, then compare seam error and diagonal-shift equivariance.
Usage: python ex955_seam.py f | trig | eval   [smoke]   (run from anywhere)"""
import sys, os, time, math, importlib.util
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
LABS = "../../../ch09/labs/"; OUT = "../../../ch09/solutions/b/"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ns, fno = load("ns", LABS + "95_ns_solver.py"), load("fno", LABS + "95_fno2d.py")
torch.set_num_threads(1)
MODE = sys.argv[1]; SMOKE = len(sys.argv) > 2
EPOCHS, BATCH, LR, NTRAIN = 4, 8, 2e-3, 200


class FNO2dIn(fno.FNO2d):
    """FNO2d whose position channels are chosen by `mode` ('xy' = the Lab model)."""
    def __init__(self, mode, **kw):
        super().__init__(**kw)
        self.mode = mode
        if mode != "xy":
            self.lift = nn.Linear(1 + {"f": 1, "trig": 4}[mode], self.width)

    def position(self, B, N1, N2, dtype):
        x = torch.arange(N1, dtype=dtype).view(1, N1, 1, 1) / N1
        y = torch.arange(N2, dtype=dtype).view(1, 1, N2, 1) / N2
        tp = 2 * math.pi
        if self.mode == "xy":
            ch = [x.expand(1, N1, N2, 1), y.expand(1, N1, N2, 1)]
        elif self.mode == "f":                                 # f / 0.1 = sin 2pi(x+y) + cos 2pi(x+y)
            ch = [torch.sin(tp * (x + y)) + torch.cos(tp * (x + y))]
        else:
            ch = [torch.sin(tp * x).expand(1, N1, N2, 1), torch.cos(tp * x).expand(1, N1, N2, 1),
                  torch.sin(tp * y).expand(1, N1, N2, 1), torch.cos(tp * y).expand(1, N1, N2, 1)]
        return torch.cat(ch, -1).expand(B, N1, N2, -1)

    def increment(self, u):
        B, N1, N2 = u.shape
        v = self.lift(torch.cat([u.unsqueeze(-1), self.position(B, N1, N2, u.dtype)], dim=-1))
        for l in range(self.n_layers):
            v = self.spec[l](v) + self.lin[l](v)
            if l < self.n_layers - 1:
                v = F.gelu(v)
        return self.proj2(F.gelu(self.proj1(v))).squeeze(-1)


d = np.load(LABS + "../data/95_ns_nu1e-3_64.npz")
w = d["w"]; T = w.shape[1] - 1
train, test = w[:NTRAIN], w[NTRAIN:]
sigma = float(train.std())

if MODE in ("f", "trig"):
    torch.manual_seed(0); np.random.seed(0)
    if SMOKE: train = train[:8]; EPOCHS = 1
    x_tr = torch.from_numpy(train[:, :-1].reshape(-1, 64, 64) / sigma)
    y_tr = torch.from_numpy(train[:, 1:].reshape(-1, 64, 64) / sigma)
    model = FNO2dIn(MODE, modes=12, width=20, n_layers=4, proj=64)
    print(f"[{MODE}] {fno.n_params(model):,d} real parameters, {len(x_tr)} pairs", flush=True)
    steps_total = EPOCHS * (len(x_tr) // BATCH)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps_total, pct_start=0.1)
    c0 = time.process_time()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(x_tr)); run = 0.0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = fno.rel_l2_loss(model(x_tr[idx]), y_tr[idx]); loss.backward(); opt.step(); sched.step()
            run += loss.item()
        print(f"[{MODE}] epoch {ep + 1}: train loss {run / (len(perm) // BATCH):.4f} | CPU {time.process_time() - c0:.0f} s", flush=True)
    torch.save({"state_dict": model.state_dict(), "cpu": time.process_time() - c0}, OUT + f"ex955_fno_{MODE}{'_smoke' if SMOKE else ''}.pt")
    sys.exit()

# ---------------------------------------------------------------- evaluation of the three models and the solver
models = {}
ck = torch.load(LABS + "../data/95_fno_ns.pt")
m = FNO2dIn("xy", **ck["config"]); m.load_state_dict(ck["state_dict"]); models["xy (Lab 9.17)"] = m.eval()
for mode in ("f", "trig"):
    ck = torch.load(OUT + f"ex955_fno_{mode}{'_smoke' if SMOKE else ''}.pt")
    m = FNO2dIn(mode, modes=12, width=20, n_layers=4, proj=64); m.load_state_dict(ck["state_dict"]); models[mode] = m.eval()
    print(f"[{mode}] training CPU time {ck['cpu']:.0f} s")
fx = ns.forcing(64); shift = lambda a: np.roll(a, (8, -8), axis=(-2, -1))
print(f"forcing invariant under the diagonal shift by 8 points: max |Sf - f| = {np.abs(shift(fx) - fx).max():.1e}")
edge = np.zeros((64, 64), bool); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
G = lambda model, a: torch.cat([model(torch.from_numpy(a[i:i + 40])) for i in range(0, len(a), 40)]).numpy()
x_eq = (test[:, 1:-1:3].reshape(-1, 64, 64) / sigma).astype(np.float32)        # 40 x 7 = 280 true states
print("  input        params    one-step   rollout t=20 (s.e.)   seam ratio (ring / rest)   max|<w>|    ||G(Sw)-SG(w)||/||G(w)||")
with torch.no_grad():
    for name, model in models.items():
        x = torch.from_numpy(test[:, :-1].reshape(-1, 64, 64) / sigma)
        p1 = torch.cat([model(x[i:i + 40]) for i in range(0, len(x), 40)]).numpy().reshape(40, T, 64, 64) * sigma
        e1 = ns.rel_l2(p1, test[:, 1:]).mean()
        out = [torch.from_numpy(test[:, 0] / sigma)]
        for _ in range(T): out.append(model(out[-1]))
        pred = torch.stack(out, 1).numpy() * sigma
        e20 = ns.rel_l2(pred[:, -1], test[:, -1])
        ae = np.abs(pred[:, -1] - test[:, -1]).mean(0)
        g = G(model, x_eq); gs = G(model, shift(x_eq).copy())
        eq = (np.sqrt(((gs - shift(g)) ** 2).sum((1, 2))) / np.sqrt((g ** 2).sum((1, 2)))).mean()
        print(f"  {name:12s} {fno.n_params(model):,d}   {e1:.4f}     {e20.mean():.4f} ({e20.std() / np.sqrt(40):.4f})       "
              f"{ae[edge].mean():.4f} / {ae[~edge].mean():.4f} = {ae[edge].mean() / ae[~edge].mean():.2f}        {np.abs(pred.mean((2, 3))).max():.2e}    {eq:.2e}")
sol = ns.NS2D(64, dt=1e-2); xs = (x_eq[::35] * sigma).astype(float)
g = sol.advance(xs, 100); gs = sol.advance(shift(xs), 100)
print(f"  solver (64^2, dt = 1e-2, 100 steps), {len(xs)} states: ||G(Sw)-SG(w)||/||G(w)|| = "
      f"{(np.sqrt(((gs - shift(g)) ** 2).sum((1, 2))) / np.sqrt((g ** 2).sum((1, 2)))).mean():.1e}")
