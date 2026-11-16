"""Lab 7.2.2 -- (A) gradient flow through time in a vanilla RNN vs LSTM;
(B) forecasting synthetic battery capacity-fade curves: persistence, linear extrapolation,
MLP and LSTM on the same 50-cycle window, horizon 100 cycles.
The fade model is synthetic (sqrt-SEI + linear + quadratic 'knee' term) -- not measured data.
Run: python 72a_lstm.py   (about 4-5 min on one shared core)"""
import time, numpy as np, torch
torch.set_num_threads(1)
T0 = time.time()

# ---------------- Part A: d h_T / d x_t versus lag T - t at initialization ----------------
print("=== Part A: ||d h_T / d x_t|| vs lag, hidden 32, input 1, T=200, mean of 10 random inits ===")
T, H = 200, 32
def grad_vs_lag(kind, seed, fbias=0.0):
    torch.manual_seed(seed)
    if kind == "RNN": net = torch.nn.RNN(1, H, batch_first=True, nonlinearity="tanh")
    else:
        net = torch.nn.LSTM(1, H, batch_first=True)
        with torch.no_grad():                         # gate order in PyTorch: input, forget, cell, output
            for name, p in net.named_parameters():
                if "bias" in name: p[H:2 * H].fill_(fbias / 2)   # two bias vectors -> total forget bias = fbias
    x = torch.randn(1, T, 1, requires_grad=True)
    out, _ = net(x)
    g, = torch.autograd.grad(out[0, -1].sum(), x)
    return g[0, :, 0].abs().numpy()[::-1]             # index = lag
lags = [1, 10, 25, 50, 100, 199]
GA = {}
for kind, fb, lab in [("RNN", 0, "vanilla RNN (tanh)"), ("LSTM", 0.0, "LSTM, forget bias 0"), ("LSTM", 3.0, "LSTM, forget bias +3")]:
    G = np.mean([grad_vs_lag(kind, s, fb) for s in range(10)], 0); GA[lab] = G
    print(f"{lab:22s} " + "  ".join(f"lag {l:3d}: {G[l]:8.1e}" for l in lags))

# ---------------- Part B: capacity-fade forecasting ----------------
def cell(rng, n=1200):
    a, b = rng.uniform(1e-3, 3e-3), rng.uniform(2e-5, 6e-5)
    nk, d = rng.uniform(400, 900), rng.uniform(1e-6, 4e-6)
    c = np.arange(n, dtype=float)
    q = 1 - a * np.sqrt(c) - b * c - d * np.clip(c - nk, 0, None) ** 2
    q = q + 0.002 * rng.normal(size=n)                 # measurement noise
    end = np.argmax(q < 0.6) if (q < 0.6).any() else n # stop at 60% capacity
    return q[:end].astype(np.float32), nk
rng = np.random.default_rng(0)
cells = [cell(rng) for _ in range(200)]
train_c, test_c = cells[:150], cells[150:]
W, Hz = 50, 100
def windows(cs, stride):
    X, Y, K = [], [], []
    for q, nk in cs:
        for s in range(0, len(q) - W - Hz, stride):
            X.append(q[s:s + W]); Y.append(q[s + W:s + W + Hz] - q[s + W - 1]); K.append(s + W - 1 + Hz > nk)
    return np.array(X), np.array(Y), np.array(K)
Xtr, Ytr, _ = windows(train_c, 5); Xte, Yte, Kte = windows(test_c, 10)
print(f"=== Part B: {len(train_c)} train cells / {len(test_c)} test cells; windows train {len(Xtr)}, test {len(Xte)} "
      f"(test windows whose horizon reaches past the knee: {Kte.mean():.2f}) ===")
lens = [len(q) for q, _ in cells]; print(f"cell lengths to 60% capacity: median {int(np.median(lens))}, range {min(lens)}-{max(lens)} cycles")

def evaluate(pred):
    e = pred - Yte
    rm = lambda m: np.sqrt(np.mean(e[m] ** 2)) * 100       # in % of nominal capacity
    return rm(np.ones(len(e), bool)), rm(~Kte), rm(Kte), np.sqrt(np.mean(e ** 2, 0)) * 100
results = {}
results["persistence"] = evaluate(np.zeros_like(Yte))
t = np.arange(W) - (W - 1); hz = np.arange(1, Hz + 1)
slope = ((Xte - Xte.mean(1, keepdims=True)) * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
results["linear extrapolation"] = evaluate(slope[:, None] * hz[None, :])
# quadratic fit through the window (an honest 'physics-flavoured' simple baseline)
A = np.stack([np.ones(W), t, t ** 2], 1); coef = np.linalg.lstsq(A, Xte.T, rcond=None)[0]
fit_last = coef[0]
results["quadratic extrapolation"] = evaluate((coef[0][:, None] + coef[1][:, None] * hz + coef[2][:, None] * hz ** 2) - fit_last[:, None])

def prep(X): return torch.tensor((X - X[:, -1:]) / 0.01, dtype=torch.float32)   # window relative to last value, in % units
class MLP(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.n = torch.nn.Sequential(torch.nn.Linear(W, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                                                     torch.nn.SiLU(), torch.nn.Linear(128, Hz))
    def forward(s, x): return s.n(x)
class LSTMf(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.l = torch.nn.LSTM(1, 64, batch_first=True); s.o = torch.nn.Linear(64, Hz)
    def forward(s, x):
        h, _ = s.l(x[:, :, None]); return s.o(h[:, -1])
def fit(mk, seed, epochs=25):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); net = mk()
    Xt, Yt = prep(Xtr), torch.tensor(Ytr / 0.01)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((len(Yt) + 255) // 256))
    for ep in range(epochs):
        perm = torch.randperm(len(Yt), generator=g)
        for b in range(0, len(Yt), 256):
            j = perm[b:b + 256]; loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad(): return net(prep(Xte)).numpy() * 0.01, net
nparams = {}
for name, mk in [("MLP", MLP), ("LSTM", LSTMf)]:
    runs = []
    for seed in range(3):
        p, net = fit(mk, seed); runs.append(evaluate(p))
    nparams[name] = sum(q.numel() for q in net.parameters())
    results[name] = tuple(np.mean([r[k] for r in runs], 0) for k in range(4)) + (np.std([r[0] for r in runs]),)
    print(f"[{name} trained, 3 seeds, {time.time()-T0:.0f} s]")
print(f"{'model':24s} {'RMSE all':>9s} {'pre-knee':>9s} {'knee':>9s} {'RMSE@h=10':>10s} {'@h=100':>8s}   (% of nominal capacity)")
for k, r in results.items():
    extra = f"   +/- {r[4]:.3f} over seeds, {nparams[k]} params" if k in nparams else ""
    print(f"{k:24s} {r[0]:9.3f} {r[1]:9.3f} {r[2]:9.3f} {r[3][9]:10.3f} {r[3][-1]:8.3f}{extra}")
# one example test cell for the figure
q, nk = test_c[3]; s = int(nk) - 80
ex = {"q": q, "nk": nk, "s": s}
xw = q[s:s + W][None]
for name, mk in [("MLP", MLP), ("LSTM", LSTMf)]:
    _, net = fit(mk, 0)
    with torch.no_grad(): ex[name] = net(prep(xw)).numpy()[0] * 0.01 + q[s + W - 1]
tt = np.arange(W) - (W - 1); sl = np.polyfit(tt, q[s:s + W], 1)[0]; ex["linear"] = q[s + W - 1] + sl * hz
np.savez("72a_lstm.npz", lagG_rnn=GA["vanilla RNN (tanh)"], lagG_lstm0=GA["LSTM, forget bias 0"], lagG_lstm3=GA["LSTM, forget bias +3"],
         names=np.array(list(results)), byh=np.array([results[k][3] for k in results]), **{f"ex_{k}": v for k, v in ex.items()})
print(f"wall {time.time()-T0:.0f} s")
