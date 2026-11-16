"""Ex 8.6.5 (reduced): DRO with hidden 128, authors' 'oi' loss, d = 2, batch 32; checkpoints -> regret@50 and
blindfold gap on 150 test landscapes (K=6, c=0.3, d=2)."""
import time, numpy as np, torch
from common import lab
torch.set_num_threads(1)
M = lab("86_dro"); M.D = 2; d = 2
t0 = time.time()
rng = np.random.default_rng(99)
G = M.sample_gmm(150, rng, d=d); x0 = M.x0_batch(150, rng)
from scipy.optimize import minimize
def gmax2(G):
    g = np.linspace(0, 1, 101); grid = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, d); F = []
    for i in range(len(G["w"])):
        v = M.f_np(G, grid[None].copy(), idx=slice(i, i + 1))[0]; best = v.max()
        for j in np.argsort(v)[-5:]:
            r = minimize(lambda x: -M.f_np(G, x[None], idx=slice(i, i + 1))[0], grid[j], method="L-BFGS-B", bounds=[(0, 1)] * d)
            best = max(best, -r.fun)
        F.append(best)
    return np.array(F)
Fs = gmax2(G)
U = np.random.default_rng(5).uniform(0, 1, (150, 50, d))
Yr = np.concatenate([M.f_np(G, x0)[:, None], M.f_np(G, U)], 1)
rr = Fs[:, None] - np.maximum.accumulate(Yr, 1)
print(f"random search (d=2): regret@10 {rr[:,10].mean():.3f} @50 {rr[:,50].mean():.3f}  (setup {time.time()-t0:.0f} s)", flush=True)

def roll(net, blind, seed=0, stochastic=True):
    gen = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(x0, dtype=torch.float32); state = None; ys, xs = [], []
    with torch.no_grad():
        for t in range(51):
            y = M.f_torch(G, x); ys.append(y); xs.append(x)
            if t == 50: break
            mean, L, state = net.step(x, torch.full_like(y, 0.5) if blind else y, state)
            x = (mean + (L @ torch.randn(x.shape, generator=gen)[..., None])[..., 0]) if stochastic else mean
            x = x.clamp(0, 1)
    Y = torch.stack(ys, 1).numpy(); X = torch.stack(xs, 1).numpy()
    reg = Fs[:, None] - np.maximum.accumulate(Y, 1)
    step = np.linalg.norm(np.diff(X, axis=1), axis=-1).mean()
    return reg, step

torch.manual_seed(0); trng = np.random.default_rng(0)
net = M.DRO(d=d, h=128); opt = torch.optim.Adam(net.parameters(), 1e-3)
CK = [0, 150, 300, 600, 1200, 1800, 2400]
for it in range(1, CK[-1] + 1):
    Gt = M.sample_gmm(32, trng, d=d)
    Y = M.run_dro(net, Gt, M.x0_batch(32, trng), M.f_torch)
    loss = -(Y[:, 1:] - torch.cummax(Y, 1).values[:, :-1]).sum(1).mean()
    opt.zero_grad(); loss.backward()
    for p in net.parameters(): p.grad.clamp_(-0.1, 0.1)
    opt.step()
    if it in CK:
        r, st = roll(net, False); rb, stb = roll(net, True); rd, _ = roll(net, False, stochastic=False)
        print(f"iter {it:5d} loss {float(loss):7.3f} | regret@10 {r[:,10].mean():.3f} @50 {r[:,50].mean():.3f}+/-{r[:,50].std(ddof=1)/np.sqrt(150):.3f}"
              f" | blind @50 {rb[:,50].mean():.3f} (diff {rb[:,50].mean()-r[:,50].mean():+.3f}) | determ @50 {rd[:,50].mean():.3f}"
              f" | mean step {st:.3f} (blind {stb:.3f}) | {time.time()-t0:.0f} s", flush=True)
