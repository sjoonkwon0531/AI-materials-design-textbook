"""86_dro_probe.py -- does the trained DRO-bsf policy actually use the observed yields?
Feeds the policy (i) the true yields, (ii) a constant 0.5 instead of every yield ("blindfolded"),
on the same 150 in-distribution test landscapes and the same sampling noise; compares best-so-far regret
and the mean distance of each query from the previous best point."""
import os, sys, importlib, numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("86_dro")
st = torch.load(os.path.join(HERE, "86_dro.pt"))
net = M.DRO(); net.load_state_dict(st["bsf"]); net.eval()
rng = np.random.default_rng(99)
G = M.sample_gmm(150, rng); Fs = M.global_max(G); x0 = M.x0_batch(150, rng)

def roll(blind, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(x0, dtype=torch.float32); state = None; ys, xs = [], []
    with torch.no_grad():
        for t in range(M.HORIZON + 1):
            y = M.f_torch(G, x); ys.append(y); xs.append(x)
            if t == M.HORIZON: break
            mean, L, state = net.step(x, torch.full_like(y, 0.5) if blind else y, state)
            x = (mean + (L @ torch.randn(x.shape, generator=gen)[..., None])[..., 0]).clamp(0, 1)
    return torch.stack(ys, 1).numpy(), torch.stack(xs, 1).numpy()

for blind in (False, True):
    Y, X = roll(blind)
    reg = Fs[:, None] - np.maximum.accumulate(Y, 1)
    ib = np.argmax(np.maximum.accumulate(Y, 1) == Y, 1)
    # distance from each query to the best point found before it
    d = []
    for i in range(len(Y)):
        for t in range(1, Y.shape[1]):
            b = np.argmax(Y[i, :t]); d.append(np.linalg.norm(X[i, t] - X[i, b]))
    print(f"{'blindfolded (y := 0.5)' if blind else 'sees true yields      '}: regret @10 {reg[:, 10].mean():.3f} "
          f"@20 {reg[:, 20].mean():.3f} @50 {reg[:, 50].mean():.3f} | mean distance of query to incumbent {np.mean(d):.3f}")
U = np.random.default_rng(5).uniform(0, 1, (20000, 2, 3))
print(f"reference: mean distance between two uniform points in the unit cube {np.linalg.norm(U[:, 0] - U[:, 1], axis=1).mean():.3f}")
