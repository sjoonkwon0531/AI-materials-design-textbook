"""Ex 8.5.6 (reduced): model-based RL from the 400 historical runs of Lab 8.15, re-simulated with per-step
film observations and per-step rewards.  5-member Gaussian MLP ensemble for (obs, a) -> (d log w's, r);
PPO (same hyper-parameters as Lab 8.14) trained inside the model; evaluated on the true simulator."""
import time, numpy as np, torch, torch.nn as nn
from common import lab
torch.set_num_threads(1)
E = lab("85_film_env"); P = E.FilmParams(); t0 = time.time()

def record(sched, seed, n=1, closed=None):
    """same RNG order as rollout_batch; returns obs (n,N+1,5), actions (n,N), rewards (n,N), J."""
    rng = np.random.default_rng(seed); d = rng.uniform(-P.delta, P.delta, n)
    c = np.zeros((n, len(P.m))); last = np.zeros(n); O, A, R = [], [], []; wprev = np.zeros(n)
    for t in range(P.N + 1):
        w, wl, wh = E.rough(c, P), E.rough(c, P, P.low), E.rough(c, P, ~P.low)
        o = np.stack([np.full(n, t / P.N), np.log(w + 1e-3), np.log(wl + 1e-3), np.log(wh + 1e-3), last], 1); O.append(o)
        if t == P.N: break
        a = np.clip(sched(t, o), -1, 1)
        T = E.to_T(a) + d + P.jitter * rng.standard_normal(n); c = E.ou_step(c, T, P, rng)
        wn = E.rough(c, P); R.append(-(wn - wprev) - P.lam * P.tb(T)); wprev = wn; last = a; A.append(a)
    return np.stack(O, 1), np.stack(A, 1), np.stack(R, 1)

# ----- the 400 historical runs (identical schedules and seeds as Lab 8.15)
rng = np.random.default_rng(0)
X = (rng.uniform(650, 720, 400)[:, None] + rng.normal(0, 10, (400, P.N)) - 700) / 100
D = [record(lambda t, o, s=s: np.full(1, s[t]), 10_000 + i) for i, s in enumerate(X)]
O = np.concatenate([d[0] for d in D]); A = np.concatenate([d[1] for d in D]); Rw = np.concatenate([d[2] for d in D])
print(f"check: label J of run 0 = {Rw[0].sum():.4f} (Lab 8.15 label computed by rollout_batch: "
      f"{E.rollout_batch(lambda t, o: np.full(1, X[0][t]), 1, 10_000, P)[0][0]:.4f})")
inp = np.concatenate([O[:, :-1], A[..., None]], -1).reshape(-1, 6)
out = np.concatenate([O[:, 1:, 1:4] - O[:, :-1, 1:4], Rw[..., None]], -1).reshape(-1, 4)
im, isd = inp.mean(0), inp.std(0) + 1e-6; om, osd = out.mean(0), out.std(0) + 1e-6
print(f"transitions {len(inp)}; data action range [{A.min():.2f}, {A.max():.2f}] (T {E.to_T(A.min()):.0f}-{E.to_T(A.max()):.0f} K)")

class GNet(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 8))
    def forward(s, x):
        o = s.f(x); return o[:, :4], o[:, 4:].clamp(-8, 3)
Xi = torch.tensor((inp - im) / isd, dtype=torch.float32); Yo = torch.tensor((out - om) / osd, dtype=torch.float32)
ens = []
for k in range(5):
    torch.manual_seed(k); net = GNet(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(k); idx = torch.randint(0, len(Xi), (len(Xi),), generator=g)   # bootstrap
    for ep in range(60):
        perm = idx[torch.randperm(len(idx), generator=g)]
        for s in range(0, len(perm), 128):
            i = perm[s:s + 128]; mu, lv = net(Xi[i])
            loss = (((mu - Yo[i]) ** 2) * torch.exp(-lv) + lv).mean(); opt.zero_grad(); loss.backward(); opt.step()
    ens.append(net.eval())
print(f"ensemble trained ({time.time()-t0:.0f} s)")
IM, ISD, OM, OSD = (torch.tensor(v, dtype=torch.float32) for v in (im, isd, om, osd))

def model_step(o, a, gen, member=None, pen=0.0):
    x = (torch.cat([o, a[:, None]], 1) - IM) / ISD
    with torch.no_grad():
        outs = [n(x) for n in ens]; mus = torch.stack([m for m, _ in outs]); lvs = torch.stack([l for _, l in outs])
    k = torch.randint(0, 5, (len(o),), generator=gen) if member is None else torch.full((len(o),), member)
    mu = mus[k, torch.arange(len(o))]; sd = torch.exp(0.5 * lvs[k, torch.arange(len(o))])
    y = (mu + sd * torch.randn(mu.shape, generator=gen)) * OSD + OM
    unc = (mus * OSD + OM).std(0)[:, 0]                       # disagreement on d log w
    o2 = o.clone(); o2[:, 0] += 1 / P.N; o2[:, 1:4] += y[:, :3]; o2[:, 4] = a
    return o2, y[:, 3] - pen * unc, unc

class AC(nn.Module):
    def __init__(s, h=64):
        super().__init__()
        s.pi = nn.Sequential(nn.Linear(5, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        s.v = nn.Sequential(nn.Linear(5, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        s.log_std = nn.Parameter(torch.full((1,), -0.7)); nn.init.zeros_(s.pi[-1].bias); s.pi[-1].weight.data.mul_(0.01)
    def dist(s, x): return torch.distributions.Normal(s.pi(x), s.log_std.exp())

o_init = torch.tensor(O[0, 0], dtype=torch.float32)
def ppo_model(seed, pen=0.0, total=120_000, nep=64, epochs=8, mb=256, lr=3e-4, lam=0.95, clip=0.2):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); ac = AC(); opt = torch.optim.Adam(ac.parameters(), lr)
    for u in range(total // (nep * P.N)):
        o = o_init.repeat(nep, 1); Os, As, LPs, Rs, Vs = [], [], [], [], []
        for t in range(P.N):
            with torch.no_grad():
                d = ac.dist(o); a = d.sample(); lp = d.log_prob(a).sum(-1); v = ac.v(o).squeeze(-1)
            ac_ = a[:, 0].clamp(-1, 1); o2, r, _ = model_step(o, ac_, gen, pen=pen)
            Os.append(o); As.append(a); LPs.append(lp); Rs.append(r); Vs.append(v); o = o2
        adv = torch.zeros(P.N, nep); g = torch.zeros(nep)
        for t in reversed(range(P.N)):
            vn = Vs[t + 1] if t < P.N - 1 else torch.zeros(nep)
            dl = Rs[t] + vn - Vs[t]; g = dl + lam * g; adv[t] = g
        ret = (adv + torch.stack(Vs)).reshape(-1); ADV = adv.reshape(-1); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        O_, A_, LP_ = torch.cat(Os), torch.cat(As), torch.cat(LPs)
        for _ in range(epochs):
            perm = torch.randperm(len(ADV))
            for s in range(0, len(ADV), mb):
                i = perm[s:s + mb]; d = ac.dist(O_[i]); ratio = torch.exp(d.log_prob(A_[i]).sum(-1) - LP_[i])
                l = -torch.min(ratio * ADV[i], ratio.clamp(1 - clip, 1 + clip) * ADV[i]).mean() + 0.5 * ((ac.v(O_[i]).squeeze(-1) - ret[i]) ** 2).mean()
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
    return ac

def pol(ac):
    return lambda t, o: ac.pi(torch.as_tensor(o, dtype=torch.float32)).squeeze(-1).detach().numpy()

def model_return(ac, n=2000, seed=1, pen=0.0):
    gen = torch.Generator().manual_seed(seed); o = o_init.repeat(n, 1); J = torch.zeros(n)
    for t in range(P.N):
        with torch.no_grad(): a = ac.pi(o).squeeze(-1).clamp(-1, 1)
        o, r, _ = model_step(o, a, gen); J += r
    return J.mean().item()

def compounding(ac, n=500):
    """real trajectories under the policy; replay the same actions through the model (mean member, no noise)
    from the true initial obs, vs one-step predictions from the TRUE previous obs."""
    Ob, Ac, Rr = record(pol(ac), 4321, n)
    o = torch.tensor(Ob[:, 0], dtype=torch.float32); err_multi, err_one = [], []
    for t in range(P.N):
        a = torch.tensor(Ac[:, t], dtype=torch.float32)
        x = (torch.cat([o, a[:, None]], 1) - IM) / ISD
        xt = (torch.cat([torch.tensor(Ob[:, t], dtype=torch.float32), a[:, None]], 1) - IM) / ISD
        with torch.no_grad():
            y = torch.stack([m(x)[0] for m in ens]).mean(0) * OSD + OM; y1 = torch.stack([m(xt)[0] for m in ens]).mean(0) * OSD + OM
        o = o.clone(); o[:, 0] += 1 / P.N; o[:, 1:4] += y[:, :3]; o[:, 4] = a
        err_multi.append(np.abs(o[:, 1].numpy() - Ob[:, t + 1, 1]).mean())
        err_one.append(np.abs(Ob[:, t, 1] + y1[:, 0].numpy() - Ob[:, t + 1, 1]).mean())
    return np.array(err_one), np.array(err_multi), Ac

for name, pen in (("MBPO-style (random member)", 0.0), ("+ disagreement penalty beta=1", 1.0)):
    for seed in (1,):
        ac = ppo_model(seed, pen=pen)
        J, w, TB = E.rollout_batch(pol(ac), 2000, 12345, P)
        mJ = model_return(ac)
        e1, em, Ac = compounding(ac)
        print(f"{name} seed {seed}: model return {mJ:7.3f} | TRUE J {J.mean():7.3f} +/- {J.std(ddof=1)/np.sqrt(2000):.3f} (w_N {w.mean():.3f}, TB {TB.mean():.2f}) | "
              f"gap {mJ - J.mean():+.3f} | set-points {E.to_T(Ac.min()):.0f}-{E.to_T(Ac.max()):.0f} K, mean by step "
              + " ".join(f"{E.to_T(x):.0f}" for x in Ac.mean(0)) + f" | {time.time()-t0:.0f} s", flush=True)
        print(f"      |log w| error vs step t=1..12: one-step " + " ".join(f"{x:.3f}" for x in e1) + " | open-loop replay " + " ".join(f"{x:.3f}" for x in em), flush=True)
