"""86_dro.py -- Ch.8 Sec.8.6 Case study: a small-scale re-implementation of the Deep Reaction Optimizer (DRO)
of Zhou, Li & Zare (2017), following the authors' public TensorFlow code (github.com/lightingghost/chemopt):

  * landscapes: reactions.GMM -- 6 Gaussians in d = 3 with means ~ U(0.01, 0.99), per-axis widths
    ~ TruncNormal(c, c/5) with c = norm_cov = 0.3 (config.json), signed weights ~ N(0, 0.2), rescaled so that
    the largest/smallest single-mode heights map to 1/0;
  * policy: rnn.StochasticRNNCell -- 2-layer LSTM, input (x_t, y_t tiled d times), output mean (d) and a d x d
    matrix -> x_{t+1} = mean + tril(M) xi  (reparameterised Gaussian sample);
  * loss: model.py 'oi' (observed improvement), direction max:  L = -sum_t [ f(x_t) - max_{s<t} f(x_s) ],
    back-propagated THROUGH the differentiable simulated landscape (learning-to-learn style), Adam lr 1e-3,
    element-wise gradient clipping to +/-0.1, batch 128 fresh landscapes per iteration, unroll 50 steps.
Deviations for a 2-core CPU budget: hidden size 64 (authors: 128), 1,500 iterations (authors: up to 50,000),
queries clipped to the unit cube [0, 1]^3 with a straight-through gradient (authors' code: optional hard clipping
to [0.01, 0.99], off by default),
mixture weights drawn per landscape.
Baselines: Nelder-Mead (scipy), CMA-ES (pycma), uniform random search, and a small GP-EI (Ch.12 preview).
Run: python 86_dro.py   (~6-8 min single thread)
"""
import os, sys, time, warnings
import numpy as np, torch, torch.nn as nn
from scipy.optimize import minimize
from scipy.stats import truncnorm, norm
import cma

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
D, K_TR, C_TR, HORIZON = 3, 6, 0.3, 50
t_start = time.time()


# ------------------------------------------------------------------ landscapes
def sample_gmm(n, rng, K=K_TR, c=C_TR, d=D):
    m = rng.uniform(0.01, 0.99, (n, K, d))
    s = c + (c / 5) * truncnorm.rvs(-2, 2, size=(n, K, d), random_state=rng)
    w = rng.normal(0, 0.2, (n, K))
    modes = w / s.prod(-1)
    return dict(m=m, s=s, w=w, top=modes.max(1), bot=modes.min(1))

def f_np(G, x, idx=None):
    """x: (n, d) one point per landscape (or (n, q, d) q points per landscape)."""
    sl = slice(None) if idx is None else idx
    m, s, w, top, bot = (G[k][sl] for k in ("m", "s", "w", "top", "bot"))
    x = x[:, None, :] if x.ndim == 2 else x                      # (n, q, d)
    z = ((x[:, :, None, :] - m[:, None]) / s[:, None]) ** 2      # (n, q, K, d)
    val = ((w / s.prod(-1))[:, None, :] * np.exp(-0.5 * z.sum(-1))).sum(-1)
    out = (val - bot[:, None]) / (top - bot)[:, None]
    return out[:, 0] if out.shape[1] == 1 else out

def f_torch(G, x):
    m, s, w = (torch.as_tensor(G[k], dtype=torch.float32) for k in ("m", "s", "w"))
    top, bot = torch.as_tensor(G["top"], dtype=torch.float32), torch.as_tensor(G["bot"], dtype=torch.float32)
    z = (((x[:, None, :] - m) / s) ** 2).sum(-1)
    return (((w / s.prod(-1)) * torch.exp(-0.5 * z)).sum(-1) - bot) / (top - bot)

def global_max(G):
    """Dense 31^3 grid on the unit cube, then L-BFGS-B polish from the 5 best grid points."""
    g = np.linspace(0, 1, 31); grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, D)
    n = len(G["w"]); F = np.empty(n)
    for i in range(n):
        v = f_np(G, np.broadcast_to(grid, (1,) + grid.shape).copy(), idx=slice(i, i + 1))[0]
        best = v.max()
        for j in np.argsort(v)[-5:]:
            r = minimize(lambda x: -f_np(G, x[None], idx=slice(i, i + 1))[0], grid[j],
                         method="L-BFGS-B", bounds=[(0, 1)] * D)
            best = max(best, -r.fun)
        F[i] = best
    return F


# ------------------------------------------------------------------ the DRO policy
class DRO(nn.Module):
    def __init__(self, d=D, h=64, layers=2):
        super().__init__()
        self.d = d; self.lstm = nn.LSTM(2 * d, h, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(h, d + d * d)
    def step(self, x, y, state):
        inp = torch.cat([x, y[:, None].expand(-1, self.d)], -1)[:, None, :]
        out, state = self.lstm(inp, state)
        o = self.proj(out[:, 0]); mean, M = o[:, :self.d], o[:, self.d:].reshape(-1, self.d, self.d)
        return mean, torch.tril(M), state

def x0_batch(n, rng):
    return np.clip(rng.normal(0.5, 0.2, (n, D)), 0.1, 0.9)

def run_dro(net, G, x0, f, steps=HORIZON, stochastic=True, gen=None, noise=0.0, rng=None):
    """Roll the policy on a batch of landscapes; returns the queried TRUE values (n, steps+1)."""
    x = torch.as_tensor(x0, dtype=torch.float32); state = None; ys = []
    for t in range(steps + 1):
        y = f(G, x)
        ys.append(y)
        y_obs = y if noise == 0 else y + torch.as_tensor(noise * rng.standard_normal(len(y)), dtype=torch.float32)
        if t == steps: break
        mean, L, state = net.step(x, y_obs, state)
        if stochastic:
            xi = torch.randn(x.shape, generator=gen) if gen is not None else torch.randn_like(x)
            x = mean + (L @ xi[..., None])[..., 0]
        else:
            x = mean
        x = x + (x.clamp(0.0, 1.0) - x).detach()     # straight-through clamp: query inside the cube,
    return torch.stack(ys, 1)                         # gradient passes as if unclamped

def train_dro(iters=1500, batch=128, seed=0, log_every=250, loss_type="oi", lr=1e-3, hidden=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = DRO(h=hidden); opt = torch.optim.Adam(net.parameters(), lr=lr); log = []
    for it in range(iters):
        G = sample_gmm(batch, rng)
        Y = run_dro(net, G, x0_batch(batch, rng), f_torch)               # (batch, 51)
        best_prev = torch.cummax(Y, 1).values[:, :-1]
        if loss_type == "oi":
            loss = -(Y[:, 1:] - best_prev).sum(1).mean()                 # 'oi' loss of the authors' code
        else:                                                            # 'bsf': -sum_t (best so far at t)
            loss = -torch.cummax(Y, 1).values[:, 1:].sum(1).mean()
        opt.zero_grad(); loss.backward()
        for p in net.parameters(): p.grad.clamp_(-0.1, 0.1)               # element-wise clipping as in model.py
        opt.step()
        if (it + 1) % log_every == 0:
            log.append((it + 1, float(loss), float(Y.max(1).values.mean())))
            print(f"    iter {it + 1:5d}  loss {float(loss):8.3f}  mean best-of-50 value {log[-1][2]:.3f}"
                  f"  ({time.time() - t_start:.0f} s)", flush=True)
    return net, log


# ------------------------------------------------------------------ baselines (one landscape at a time)
def trace_of(fun, budget=HORIZON + 1):
    vals = []
    def wrapped(x):
        v = fun(np.clip(x, 0, 1)); vals.append(v); return -v
    return wrapped, vals

def pad(vals, budget=HORIZON + 1):
    vals = list(vals[:budget]);
    while len(vals) < budget: vals.append(vals[-1] if vals else 0.0)   # optimiser stopped: no new information
    return vals

def nelder_mead(fun, x0):
    w, vals = trace_of(fun)
    minimize(w, x0, method="Nelder-Mead", bounds=[(0, 1)] * D,
             options=dict(maxfev=HORIZON + 1, xatol=1e-4, fatol=1e-6, initial_simplex=None))
    return pad(vals)

def cmaes(fun, x0, seed):
    w, vals = trace_of(fun)
    es = cma.CMAEvolutionStrategy(list(x0), 0.2, dict(bounds=[0, 1], seed=seed + 1, verbose=-9,
                                                      maxfevals=HORIZON + 1))
    vals.append(fun(np.asarray(x0)))                       # the same first experiment as every method
    while len(vals) < HORIZON + 1:
        X = es.ask(); F = [w(x) for x in X]; es.tell(X, F)
    return pad(vals)

def random_search(fun, x0, rng):
    return [fun(x0)] + [fun(rng.uniform(0, 1, D)) for _ in range(HORIZON)]

def gp_ei(fun, x0, rng, n_cand=2000, ls=0.2, noise=1e-4):
    """Minimal GP-EI (RBF kernel, fixed length scale 0.2, unit prior variance on standardized y)."""
    X = [np.asarray(x0)]; Y = [fun(x0)]
    for t in range(HORIZON):
        Xa, Ya = np.array(X), np.array(Y); mu0, sd0 = Ya.mean(), Ya.std() + 1e-9; Yn = (Ya - mu0) / sd0
        k = lambda A, B: np.exp(-0.5 * ((A[:, None] - B[None]) ** 2).sum(-1) / ls ** 2)
        Kxx = k(Xa, Xa) + noise * np.eye(len(Xa)); Lc = np.linalg.cholesky(Kxx)
        al = np.linalg.solve(Lc.T, np.linalg.solve(Lc, Yn))
        C = rng.uniform(0, 1, (n_cand, D)); Ks = k(C, Xa)
        mu = Ks @ al; v = np.linalg.solve(Lc, Ks.T); sd = np.sqrt(np.clip(1 - (v ** 2).sum(0), 1e-12, None))
        z = (mu - Yn.max()) / sd; ei = (mu - Yn.max()) * norm.cdf(z) + sd * norm.pdf(z)
        xn = C[int(np.argmax(ei))]; X.append(xn); Y.append(fun(xn))
    return Y


# ------------------------------------------------------------------ evaluation
def evaluate(nets, G, Fstar, x0, label, noise=0.0, seed=0):
    n = len(Fstar); out = {}
    with torch.no_grad():
        for nm, net in nets.items():
            gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
            out[f"{nm} (stochastic)"] = run_dro(net, G, x0, f_torch, gen=gen, noise=noise, rng=rng).numpy()
            if nm == "DRO-bsf":
                rng = np.random.default_rng(seed)
                out[f"{nm} (deterministic)"] = run_dro(net, G, x0, f_torch, stochastic=False,
                                                       noise=noise, rng=rng).numpy()
    for name in ("Nelder-Mead", "CMA-ES", "Random search", "GP-EI"):
        rows = []
        for i in range(n):
            nrng = np.random.default_rng(1000 + i)
            true = lambda x, i=i: float(f_np(G, np.asarray(x, float)[None], idx=slice(i, i + 1))[0])
            seen = []                                         # what the optimiser observes (possibly noisy)
            def fun(x, true=true):
                v = true(x); seen.append(v); return v + (noise * nrng.standard_normal() if noise else 0.0)
            if name == "Nelder-Mead": nelder_mead(fun, x0[i])
            elif name == "CMA-ES": cmaes(fun, x0[i], i)
            elif name == "Random search": random_search(fun, x0[i], np.random.default_rng(2000 + i))
            else: gp_ei(fun, x0[i], np.random.default_rng(3000 + i))
            rows.append(pad(seen))
        out[name] = np.array(rows)
    reg = {k: Fstar[:, None] - np.maximum.accumulate(v, 1) for k, v in out.items()}   # best-so-far regret
    print(f"\n[{label}]  {n} test landscapes; regret = f* - best TRUE value found so far (mean +/- s.e.)")
    print(f"  {'method':<22s} {'after 10':>14s} {'after 20':>14s} {'after 50':>14s} {'P(regret<0.05 @50)':>19s}")
    for k, r in reg.items():
        se = lambda c: r[:, c].std(ddof=1) / np.sqrt(n)
        print(f"  {k:<22s} {r[:, 10].mean():7.3f}+/-{se(10):.3f} {r[:, 20].mean():7.3f}+/-{se(20):.3f} "
              f"{r[:, 50].mean():7.3f}+/-{se(50):.3f} {np.mean(r[:, 50] < 0.05):19.2f}")
    return reg


if __name__ == "__main__":
    print("training DRO on fresh simulated landscapes (K = 6, c = 0.3, d = 3), batch 128")
    print("  (1) authors' default loss 'oi', 600 iterations")
    net_oi, log_oi = train_dro(iters=600, loss_type="oi", log_every=100)
    print("  (2) best-so-far loss 'bsf' (deviation, see text), 1,200 iterations")
    net_bsf, log_bsf = train_dro(iters=1200, loss_type="bsf", log_every=200)
    torch.save({"oi": net_oi.state_dict(), "bsf": net_bsf.state_dict()}, os.path.join(HERE, "86_dro.pt"))
    nets = {"DRO-oi": net_oi, "DRO-bsf": net_bsf}
    results = {}
    N_TEST = 150
    for label, K, c, noise in [("in-distribution: K=6, c=0.3", 6, 0.3, 0.0),
                               ("OOD-rugged: K=12, c=0.15", 12, 0.15, 0.0),
                               ("OOD-noisy: K=6, c=0.3, sigma_y=0.05", 6, 0.3, 0.05)]:
        rng = np.random.default_rng(99)
        G = sample_gmm(N_TEST, rng, K=K, c=c); Fs = global_max(G); x0 = x0_batch(N_TEST, rng)
        results[label] = evaluate(nets, G, Fs, x0, label, noise=noise)
        print(f"  ({time.time() - t_start:.0f} s)", flush=True)
    np.savez(os.path.join(HERE, "86_dro_results.npz"), log_oi=np.array(log_oi), log_bsf=np.array(log_bsf),
             **{f"{lab}__{k}": v for lab, r in results.items() for k, v in r.items()})
    print(f"wall {time.time() - t_start:.0f} s")
