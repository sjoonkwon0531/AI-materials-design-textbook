"""Lab 81_pg: REINFORCE (no baseline), REINFORCE + learned value baseline, and A2C (GAE) on CartPole-v1,
plus a direct measurement of policy-gradient estimator variance at a fixed policy.
From scratch in torch. torch.set_num_threads(1). Runtime ~5-7 min on one core."""
import numpy as np, torch, torch.nn as nn, gymnasium as gym, time, sys
torch.set_num_threads(1)
GAMMA, LAM = 0.99, 0.95

def mlp(out):
    return nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, out))

def np_policy(pi):
    """Copy the torch MLP into numpy for fast action sampling (same function, no autograd)."""
    Ws = [(l.weight.detach().numpy().copy(), l.bias.detach().numpy().copy()) for l in pi if isinstance(l, nn.Linear)]
    def f(s):
        h = s
        for i, (W, b) in enumerate(Ws):
            h = W @ h + b
            if i < len(Ws) - 1: h = np.tanh(h)
        return h
    return f

def rollout(env, pi, seed=None, rng=None):
    """One episode. Returns states, actions, rewards, terminated flag, last state."""
    f = np_policy(pi); rng = rng or np.random.default_rng(seed)
    s, _ = env.reset(seed=seed); S, A, Rw = [], [], []
    while True:
        z = f(s); p1 = 1.0 / (1.0 + np.exp(z[0] - z[1]))          # softmax over 2 actions
        a = int(rng.random() < p1)
        s2, r, term, trunc, _ = env.step(a)
        S.append(s); A.append(a); Rw.append(r); s = s2
        if term or trunc:
            return (torch.as_tensor(np.array(S), dtype=torch.float32), torch.as_tensor(A),
                    np.array(Rw, dtype=np.float32), term, torch.as_tensor(s2, dtype=torch.float32))

def reward_to_go(rw, gamma=GAMMA, bootstrap=0.0):
    G = np.zeros_like(rw); g = bootstrap
    for t in range(len(rw) - 1, -1, -1):
        g = rw[t] + gamma * g; G[t] = g
    return G

def gae(rw, v, v_last, lam=LAM, gamma=GAMMA):
    """Generalized advantage estimation: A_t = sum_l (gamma lam)^l delta_{t+l}."""
    v_next = np.append(v[1:], v_last); delta = rw + gamma * v_next - v
    A = np.zeros_like(rw); a = 0.0
    for t in range(len(rw) - 1, -1, -1):
        a = delta[t] + gamma * lam * a; A[t] = a
    return A

def train(method, seed, n_ep=300, lr_pi=3e-3, lr_v=1e-2):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make("CartPole-v1"); env.action_space.seed(seed)
    pi, V = mlp(2), mlp(1)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=lr_pi); opt_v = torch.optim.Adam(V.parameters(), lr=lr_v)
    lengths = []
    for ep in range(n_ep):
        S, A, rw, term, s_last = rollout(env, pi, seed=seed * 10_000 + ep)
        lengths.append(len(rw))
        logp = torch.distributions.Categorical(logits=pi(S)).log_prob(A)
        if method == "reinforce":
            w = torch.as_tensor(reward_to_go(rw))
        else:
            with torch.no_grad():
                v = V(S).squeeze(-1).numpy(); v_last = 0.0 if term else float(V(s_last))
            if method == "baseline":
                G = reward_to_go(rw); w = torch.as_tensor(G - v)
                target = torch.as_tensor(G)
            else:  # "a2c": GAE advantages, critic regresses on lambda-returns
                Adv = gae(rw, v, v_last); w = torch.as_tensor(Adv)
                target = torch.as_tensor(Adv + v)
            for _ in range(5):                                     # critic: a few regression steps
                opt_v.zero_grad(); ((V(S).squeeze(-1) - target) ** 2).mean().backward(); opt_v.step()
        opt_pi.zero_grad(); (-(w.float() * logp).sum() / 100.0).backward(); opt_pi.step()
    env.close()
    return np.array(lengths), pi, V

# ---------------------------------------------------------------- gradient-variance experiment
def flat_grad(pi, S, A, w):
    pi.zero_grad()
    logp = torch.distributions.Categorical(logits=pi(S)).log_prob(A)
    (-(torch.as_tensor(w, dtype=torch.float32) * logp).sum()).backward()
    return torch.cat([p.grad.flatten() for p in pi.parameters()]).numpy().copy()

def variance_experiment(pi, n_fit=300, n_eval=300, seed=123):
    env = gym.make("CartPole-v1")
    # fit a value function for this fixed policy on independent episodes (MC regression)
    torch.manual_seed(seed); Vf = mlp(1); opt = torch.optim.Adam(Vf.parameters(), lr=1e-2)
    data = [rollout(env, pi, seed=seed * 100_000 + i) for i in range(n_fit)]
    Sx = torch.cat([d[0] for d in data]); Gy = torch.as_tensor(np.concatenate([reward_to_go(d[2]) for d in data]))
    for it in range(1500):
        opt.zero_grad(); ((Vf(Sx).squeeze(-1) - Gy) ** 2).mean().backward(); opt.step()
    mean_len = np.mean([len(d[2]) for d in data])
    eps = [rollout(env, pi, seed=seed * 100_000 + n_fit + i) for i in range(n_eval)]
    bconst = np.mean(np.concatenate([reward_to_go(e[2]) for e in eps]))
    est = {k: [] for k in ["total return G_0", "gamma^t G_t (exact)", "reward-to-go G_t", "G_t - const b",
                            "G_t - V(s_t)", "GAE lambda=0.95", "TD residual (lambda=0)"]}
    for S, A, rw, term, s_last in eps:
        G = reward_to_go(rw)
        with torch.no_grad():
            v = Vf(S).squeeze(-1).numpy(); vl = 0.0 if term else float(Vf(s_last))
        est["total return G_0"].append(flat_grad(pi, S, A, np.full_like(rw, G[0])))
        est["gamma^t G_t (exact)"].append(flat_grad(pi, S, A, GAMMA ** np.arange(len(rw)) * G))
        est["reward-to-go G_t"].append(flat_grad(pi, S, A, G))
        est["G_t - const b"].append(flat_grad(pi, S, A, G - bconst))
        est["G_t - V(s_t)"].append(flat_grad(pi, S, A, G - v))
        est["GAE lambda=0.95"].append(flat_grad(pi, S, A, gae(rw, v, vl, 0.95)))
        est["TD residual (lambda=0)"].append(flat_grad(pi, S, A, gae(rw, v, vl, 0.0)))
    env.close()
    ref = np.mean(est["reward-to-go G_t"], 0); ref0 = np.mean(est["gamma^t G_t (exact)"], 0)
    rows = []
    for k, g in est.items():
        g = np.array(g); m = g.mean(0)
        tv = g.var(0, ddof=1).sum()                         # trace of covariance (per-episode estimator)
        cos = m @ ref / np.linalg.norm(m) / np.linalg.norm(ref)
        cos0 = m @ ref0 / np.linalg.norm(m) / np.linalg.norm(ref0)
        rows.append((k, tv, np.linalg.norm(m), np.sqrt(tv / len(g)), cos0, cos))
    return mean_len, rows

if __name__ == "__main__":
    t0 = time.time()
    seeds = range(5); curves = {}
    for method in (("reinforce", "baseline", "a2c") if "varonly" not in sys.argv else ()):
        L = []
        for sd in seeds:
            lens, pi, V = train(method, sd); L.append(lens)
            if method == "baseline" and sd == 0: pi_snap = None
        L = np.array(L); curves[method] = L
        last = L[:, -50:].mean(1)
        first475 = [int(np.argmax(np.convolve(l, np.ones(20) / 20, "valid") >= 475)) + 20
                    if (np.convolve(l, np.ones(20) / 20, "valid") >= 475).any() else -1 for l in L]
        print(f"{method:9s}: mean length last 50 episodes = {last.mean():6.1f} +- {last.std(ddof=1)/np.sqrt(len(L)):5.1f}"
              f" (per seed {np.round(last).astype(int).tolist()}); first episode with 20-ep avg >= 475: {first475}"
              f"  [{time.time()-t0:.0f} s]")
        sys.stdout.flush()
    if curves: np.savez("../ch08/data/81_pg_curves.npz", **curves)

    # variance experiment at a fixed, partially trained policy (REINFORCE+baseline, seed 0, 60 episodes)
    _, pi_mid, _ = train("baseline", 0, n_ep=60)
    mean_len, rows = variance_experiment(pi_mid)
    print(f"\nGradient-estimator variance at a fixed policy (mean episode length {mean_len:.1f}), 300 episodes")
    print(f"{'estimator':24s} {'trCov':>9s} {'var/G_t':>8s} {'|mean|':>7s} {'s.e.':>6s} {'cos(exact)':>10s} {'cos(G_t)':>9s}")
    base = rows[2][1]
    for k, tv, nm, se, cos0, cos in rows:
        print(f"{k:24s} {tv:9.3g} {tv/base:8.3f} {nm:7.1f} {se:6.1f} {cos0:10.3f} {cos:9.3f}")
    np.save("../ch08/data/81_pg_variance.npy", np.array([r[1:] for r in rows]))
    print(f"runtime {time.time()-t0:.0f} s")
