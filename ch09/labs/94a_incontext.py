"""Lab 9.4-A: a prior-fitted, in-context operator versus the exact Bayesian posterior predictive.

Task prior : u_t + c u_x = nu u_xx on [0, 2 pi), T = 1, with nu ~ logU[0.02, 0.5], c ~ U[-1.5, 1.5].
             In Fourier space the solution operator is diagonal: uhat_T(k) = m_k uhat_0(k),
             m_k = exp(-nu k^2 - i c k), k = 1..8.
Context    : K pairs (u0_i, y_i), y_i = u_T,i + complex Gaussian noise (sigma = 0.2 per mode).
Query      : a new u0; predict u_T.  No (nu, c) is ever shown to the network.
Compared   : prior mean (K = 0) | per-mode least squares S_k / P_k | exact Bayes (grid posterior) | network.
Run: python 94a_incontext.py   (about 2 CPU-min)"""
import math, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
M, SIG, KTR = 8, 0.2, 8                                      # modes, noise level, max context size in training
kk = np.arange(1, M + 1); sk = 1.0 / kk                      # prior std of |uhat_0(k)|
NU_LO, NU_HI, C_MAX = 0.02, 0.5, 1.5


def sample_tasks(n, K, rng, nu=None, c=None):
    """Returns context inputs x (n, K, M), noisy outputs y (n, K, M), query xq (n, M) and true multiplier m."""
    nu = np.exp(rng.uniform(np.log(NU_LO), np.log(NU_HI), n)) if nu is None else nu
    c = rng.uniform(-C_MAX, C_MAX, n) if c is None else c
    m = np.exp(-nu[:, None] * kk ** 2 - 1j * c[:, None] * kk)                       # (n, M)
    cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
    x = cn(n, K, M) * sk; xq = cn(n, M) * sk
    y = m[:, None, :] * x + SIG * cn(n, K, M)
    return x, y, xq, m


def suff_stats(x, y):
    """S_k = sum_i y_ik conj(x_ik),  P_k = sum_i |x_ik|^2  (all the likelihood needs)."""
    return (y * x.conj()).sum(1), (np.abs(x) ** 2).sum(1)


def features(S, P):
    """Network input: ridge estimate S/(P + sigma^2) (Re, Im) and the log-precision log(1 + P / sigma^2)."""
    mh = S / (P + SIG ** 2)
    return np.concatenate([mh.real, mh.imag, np.log1p(P / SIG ** 2)], 1).astype(np.float32)


# ---- exact Bayes on a grid over (log nu, c) --------------------------------------------------------
G_NU = np.exp(np.linspace(np.log(NU_LO), np.log(NU_HI), 160)); G_C = np.linspace(-C_MAX, C_MAX, 1501)
gnu, gc = [a.ravel() for a in np.meshgrid(G_NU, G_C, indexing="ij")]
m_grid = np.exp(-gnu[:, None] * kk ** 2 - 1j * gc[:, None] * kk)                    # (G, M)
A_grid = np.concatenate([2 * m_grid.real, 2 * m_grid.imag, -np.abs(m_grid) ** 2], 1) / SIG ** 2   # (G, 3M)


def bayes_multiplier(S, P, chunk=25):
    """Posterior mean of m_k given the context: sum_g w_g m_k(g), w_g ~ prior x likelihood."""
    out = np.empty(S.shape, complex)
    for i in range(0, len(S), chunk):
        F = np.concatenate([S[i:i + chunk].real, S[i:i + chunk].imag, P[i:i + chunk]], 1)          # (b, 3M)
        ll = F @ A_grid.T                                    # log-likelihood up to a constant, (b, G)
        w = np.exp(ll - ll.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        out[i:i + chunk] = w @ m_grid
    return out


# ---- the prior-fitted network: context statistics -> Fourier multiplier R(k) -----------------------
net = nn.Sequential(nn.Linear(3 * M, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                    nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2 * M))


def net_multiplier(S, P):
    with torch.no_grad():
        o = net(torch.from_numpy(features(S, P))).numpy().astype(float)
    return o[:, :M] + 1j * o[:, M:]


if __name__ == "__main__":
    torch.manual_seed(0); rng = np.random.default_rng(0)
    steps, bs = 30000, 256
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    sk_t = torch.tensor(sk, dtype=torch.float32)
    t0 = time.process_time(); w0 = time.time()
    for s in range(steps):
        K = int(rng.integers(0, KTR + 1))                    # context size 0..8, a fresh batch of tasks each step
        x, y, xq, m = sample_tasks(bs, K, rng)
        S, P = suff_stats(x, y)
        o = net(torch.from_numpy(features(S, P)))
        R = torch.complex(o[:, :M], o[:, M:])
        xq_t, yq_t = torch.from_numpy(xq).to(torch.cfloat), torch.from_numpy(m * xq).to(torch.cfloat)
        loss = ((R * xq_t - yq_t).abs() ** 2).sum(1).mean()  # squared error on the held-out (query) label
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (s + 1) % 10000 == 0:
            print(f"step {s + 1}: loss {loss.item():.4f}", flush=True)
    print(f"prior fitting: {steps} steps x {bs} tasks = {steps * bs / 1e6:.2f} M tasks, "
          f"{time.process_time() - t0:.0f} CPU-s (wall {time.time() - w0:.0f} s), "
          f"{sum(p.numel() for p in net.parameters())} parameters")

    # ---- evaluation --------------------------------------------------------------------------------
    def evaluate(n, K, seed, nu=None, c=None):
        rg = np.random.default_rng(seed)
        x, y, xq, m = sample_tasks(n, K, rg, nu, c)
        S, P = suff_stats(x, y)
        true = m * xq
        rel = lambda mult: float(np.sqrt((np.abs(mult * xq - true) ** 2).sum() / (np.abs(true) ** 2).sum()))
        mb = bayes_multiplier(S, P); mn = net_multiplier(S, P)
        gap = float(np.sqrt((np.abs((mn - mb) * xq) ** 2).sum() / (np.abs(true) ** 2).sum()))
        ls = S / np.maximum(P, 1e-12) if K > 0 else np.zeros_like(S)
        return rel(ls), rel(mb), rel(mn), gap, (m, mb, mn, ls)

    res = {}
    print("\nin-prior test, 500 tasks (relative L2 error of the predicted u_T for a new u0):")
    print("   K   per-mode LS   exact Bayes   network   |network - Bayes|")
    for K in (0, 1, 2, 4, 8, 16, 32):
        t0 = time.process_time(); r = evaluate(500, K, 100 + K); tb = time.process_time() - t0
        res[K] = r[:4]
        tag = "" if K <= KTR else "   (K beyond the training range)"
        print(f"  {K:2d}     {r[0]:6.3f}       {r[1]:6.3f}      {r[2]:6.3f}       {r[3]:6.3f}{tag}")
    # timing: amortised inference vs grid posterior
    x, y, xq, m = sample_tasks(500, 8, np.random.default_rng(5)); S, P = suff_stats(x, y)
    t0 = time.process_time(); bayes_multiplier(S, P); tb = time.process_time() - t0
    t0 = time.process_time(); [net_multiplier(S, P) for _ in range(20)]; tn = (time.process_time() - t0) / 20
    print(f"cost per task (K = 8): grid Bayes {1e3 * tb / 500:.2f} ms ({len(gnu)} grid points), "
          f"network {1e6 * tn / 500:.1f} us  -> ratio {tb / tn:.0f}")

    print("\nout-of-prior test (K = 8, 500 tasks):")
    rg = np.random.default_rng(7)
    for name, nu, c in [("nu in [0.005, 0.015] (below the prior range)", np.exp(rg.uniform(np.log(0.005), np.log(0.015), 500)), None),
                        ("c in [2, 3] (outside the prior range)", None, rg.uniform(2, 3, 500)),
                        ("in prior (reference)", None, None)]:
        r = evaluate(500, 8, 300, nu, c)
        print(f"   {name:46s} LS {r[0]:.3f} | Bayes(grid prior) {r[1]:.3f} | network {r[2]:.3f}")

    # one task for the figure: posterior over (nu, c) for K = 1 and K = 8, and multipliers
    rg = np.random.default_rng(11); nu0, c0 = np.array([0.06]), np.array([0.9])
    fig = {}
    for K in (1, 8):
        x, y, xq, m = sample_tasks(1, K, rg, nu0, c0); S, P = suff_stats(x, y)
        F = np.concatenate([S.real, S.imag, P], 1); ll = (F @ A_grid.T)[0]
        w = np.exp(ll - ll.max()); w /= w.sum()
        fig[f"post{K}"] = w.reshape(len(G_NU), len(G_C)); fig[f"mb{K}"] = bayes_multiplier(S, P)[0]
        fig[f"mn{K}"] = net_multiplier(S, P)[0]; fig[f"ls{K}"] = (S / P)[0]
    np.savez("../ch09/data/94a_incontext.npz", Ks=np.array(list(res)), res=np.array(list(res.values())),
             G_NU=G_NU, G_C=G_C, m_true=np.exp(-nu0 * kk ** 2 - 1j * c0 * kk), **fig)
