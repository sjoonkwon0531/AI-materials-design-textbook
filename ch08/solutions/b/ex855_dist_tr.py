"""Ex 8.5.5: distance-to-data trust region, score - beta*max(0, d(x) - d_max), beta = 10, d in action units."""
import os, sys, time, numpy as np, torch
from common import lab, LABS
torch.set_num_threads(1)
src = open(os.path.join(LABS, "85_sam_surrogate.py")).read().split("# ---------------------------------------------------------------- data")[0]
sys.path.insert(0, LABS); ns = {"__file__": os.path.join(LABS, "85_sam_surrogate.py")}; exec(src, ns)
E, P, train, maximise, true_J = ns["E"], ns["P"], ns["train"], ns["maximise"], ns["true_J"]
t0 = time.time()
rng = np.random.default_rng(0); n_tr = 400
base = rng.uniform(650, 720, n_tr)[:, None] + rng.normal(0, 10, (n_tr, P.N)); X = (base - 700) / 100
y = np.array([E.rollout_batch(lambda t, o, s=s: np.full(1, s[t]), 1, 10_000 + i, P)[0][0] for i, s in enumerate(X)])
ymu, ysd = y.mean(), y.std(); yn = (y - ymu) / ysd
Xt = torch.tensor(X, dtype=torch.float32)
nn_d = np.sort(np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) + np.eye(n_tr) * 1e9, 1)[:, 0]
print(f"label sd {ysd:.3f}; NN distance between training runs (action units): median {np.median(nn_d):.3f}, max {nn_d.max():.3f}")
lo, hi = -torch.ones(P.N), torch.ones(P.N)
R = {}
for seed in range(5):
    nets = {"AdamW": train(X, yn, seed), "SAM rho=0.2": train(X, yn, seed, "sam", 0.2)}
    for nm, net in nets.items():
        for dmax in (0.2, 0.5, 1.0):
            def score(Z, net=net, dmax=dmax):
                d = torch.cdist(Z, Xt).min(1).values
                return net(Z).squeeze(-1) - 10.0 * torch.clamp(d - dmax, min=0)
            x, _ = maximise(score, lo, hi, seed)
            with torch.no_grad(): pred = float(net(torch.tensor(x[None], dtype=torch.float32))) * ysd + ymu
            tj = true_J(x); d = np.sqrt(((X - x) ** 2).sum(1)).min()
            R.setdefault((nm, dmax), []).append((pred, tj, pred - tj, d, E.to_T(x).min(), E.to_T(x).max()))
    print(f"seed {seed} done ({time.time()-t0:.0f} s)", flush=True)
print(f"{'model':<12s} {'d_max':>5s} {'pred J':>7s} {'true J':>14s} {'gap':>12s} {'dist[au]':>8s} {'T range':>12s}")
for (nm, dmax), r in R.items():
    r = np.array(r)
    print(f"{nm:<12s} {dmax:5.1f} {r[:,0].mean():7.2f} {r[:,1].mean():7.2f}+/-{r[:,1].std(ddof=1):4.2f} {r[:,2].mean():6.2f}+/-{r[:,2].std(ddof=1):4.2f} "
          f"{r[:,3].mean():8.3f} {r[:,4].mean():5.0f}-{r[:,5].mean():4.0f} K | per seed true J " + " ".join(f"{v:.2f}" for v in r[:, 1]))
