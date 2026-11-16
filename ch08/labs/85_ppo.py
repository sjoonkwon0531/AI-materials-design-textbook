"""85_ppo.py -- Ch.8 Sec.8.5 Lab (a): Gymnasium film-growth env + PPO from scratch vs tuned schedules.
Run: python 85_ppo.py   (single thread; ~5-7 min on a shared 2-core machine)
"""
import importlib, sys, os, time, json
import numpy as np, torch, torch.nn as nn
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
E = importlib.import_module("85_film_env")
P = E.FilmParams()
TEST_SEED, N_TEST = 12345, 2000
t_start = time.time()

# ---------------------------------------------------------------- 1. the environment contract
env = E.FilmGrowthEnv()
check_env(env, skip_render_check=True)              # raises on any API violation
print("check_env: passed |", env.observation_space, "|", env.action_space)
o1, _ = env.reset(seed=7); o2, _ = E.FilmGrowthEnv().reset(seed=7)
print("same seed -> same first obs:", np.array_equal(o1, o2))

# ---------------------------------------------------------------- 2. baselines (tuned on TRAIN seeds)
def const(Tset): return lambda t, o: np.full(len(o), (Tset - 700) / 100)
def J_of(fn, n=600, seed=1, **kw): return E.rollout_batch(fn, n, seed, P, **kw)[0].mean()

Ts = np.arange(620, 781, 5)
Jc = [J_of(const(T)) for T in Ts]; T_const = Ts[int(np.argmax(Jc))]
best2 = max(((J_of(lambda t, o, k=k, a=a, b=b: np.full(len(o), ((a if t < k else b) - 700) / 100), 300), (k, a, b))
             for k in (3, 6, 9) for a in range(640, 741, 10) for b in range(640, 761, 10)), key=lambda z: z[0])
k2, T1, T2 = best2[1]
# oracle that knows delta: best constant ACTUAL temperature at delta = 0, set-point corrected by -delta
Tact = Ts[int(np.argmax([J_of(const(T), delta=0.0) for T in Ts]))]
print(f"tuned constant set-point {T_const} K | tuned two-level ({k2} steps at {T1} K, then {T2} K) "
      f"| oracle actual T {Tact} K")

# random search over open-loop schedules: 2,500 candidates x 10 episodes = 25,000 episodes (= PPO budget)
rs_rng = np.random.default_rng(3)
cands = rs_rng.uniform(-1, 1, (2500, P.N)); scores = []
for i, s in enumerate(cands):
    scores.append(E.rollout_batch(lambda t, o, s=s: np.full(len(o), s[t]), 10, 100 + i, P)[0].mean())
rs_best = cands[int(np.argmax(scores))]
print(f"random search: best training score {max(scores):.3f}; schedule (K) "
      + " ".join(f"{x:.0f}" for x in E.to_T(rs_best)))

# ---------------------------------------------------------------- 3. PPO from scratch
class ActorCritic(nn.Module):
    def __init__(self, n_obs=5, h=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(n_obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.v = nn.Sequential(nn.Linear(n_obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.log_std = nn.Parameter(torch.full((1,), -0.7))
        nn.init.zeros_(self.pi[-1].bias); self.pi[-1].weight.data.mul_(0.01)
    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.exp())

def ppo(seed, open_loop=False, total_steps=240_000, n_env=16, n_roll=48, epochs=8, mb=256, lr=3e-4,
        gamma=1.0, lam=0.95, clip=0.2, ent=0.0, log_every=50):
    torch.manual_seed(seed); np.random.seed(seed)
    venv = gym.vector.SyncVectorEnv([(lambda: E.OpenLoop(E.FilmGrowthEnv())) if open_loop else (lambda: E.FilmGrowthEnv())
                                     for _ in range(n_env)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = venv.reset(seed=[1000 * seed + i for i in range(n_env)])
    ac = ActorCritic(); opt = torch.optim.Adam(ac.parameters(), lr=lr)
    n_upd = total_steps // (n_env * n_roll); curve = []; ep_ret = np.zeros(n_env); done_rets = []
    for u in range(n_upd):
        O, A, LP, R, D, V = [], [], [], [], [], []
        for t in range(n_roll):
            x = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = ac.dist(x); a = d.sample(); lp = d.log_prob(a).sum(-1); v = ac.v(x).squeeze(-1)
            obs2, r, term, trunc, info = venv.step(a.numpy())
            O.append(x); A.append(a); LP.append(lp); V.append(v)
            R.append(torch.as_tensor(r, dtype=torch.float32)); D.append(torch.as_tensor(term | trunc, dtype=torch.float32))
            ep_ret += r
            for i in np.where(term | trunc)[0]: done_rets.append(ep_ret[i]); ep_ret[i] = 0.0
            obs = obs2
        with torch.no_grad(): v_last = ac.v(torch.as_tensor(obs, dtype=torch.float32)).squeeze(-1)
        # GAE(lambda) -- Sec.8.1.5; a terminal step has no bootstrap
        adv = torch.zeros(n_roll, n_env); g = torch.zeros(n_env)
        for t in reversed(range(n_roll)):
            v_next = v_last if t == n_roll - 1 else V[t + 1]
            delta = R[t] + gamma * v_next * (1 - D[t]) - V[t]
            g = delta + gamma * lam * (1 - D[t]) * g; adv[t] = g
        ret = adv + torch.stack(V)
        O_, A_, LP_ = torch.cat(O), torch.cat(A), torch.cat(LP)
        ADV, RET = adv.reshape(-1), ret.reshape(-1)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)          # advantage normalisation
        n = len(ADV)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, mb):
                i = perm[s:s + mb]; d = ac.dist(O_[i])
                ratio = torch.exp(d.log_prob(A_[i]).sum(-1) - LP_[i])
                l_clip = -torch.min(ratio * ADV[i], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[i]).mean()
                l_v = 0.5 * ((ac.v(O_[i]).squeeze(-1) - RET[i]) ** 2).mean()
                loss = l_clip + l_v - ent * d.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        if (u + 1) % log_every == 0 or u == n_upd - 1:
            curve.append(((u + 1) * n_env * n_roll, float(np.mean(done_rets[-400:]))))
    venv.close()
    return ac, curve

def policy_fn(ac):
    def f(t, o):
        with torch.no_grad():
            return ac.pi(torch.as_tensor(o, dtype=torch.float32)).squeeze(-1).numpy()   # deterministic: mean action
    return f

# ---------------------------------------------------------------- 4. evaluation on common TEST seeds
def report(name, fn, store):
    J, w, TB, T, d = E.rollout_batch(fn, N_TEST, TEST_SEED, P, return_traj=True)
    store[name] = dict(J=J, w=w, TB=TB, T=T, d=d)
    print(f"  {name:<30s} J = {J.mean():7.3f} +/- {J.std(ddof=1) / np.sqrt(len(J)):.3f} | "
          f"w_N {w.mean():.3f} | TB {TB.mean():6.2f} | 5th pct J {np.percentile(J, 5):7.3f}")
    return J.mean()

res = {}
print(f"\nTEST: {N_TEST} episodes, seed {TEST_SEED}, delta ~ U(-{P.delta:.0f},{P.delta:.0f}) K, N = {P.N} ML")
report("tuned constant", const(T_const), res)
report("tuned two-level", lambda t, o: np.full(len(o), ((T1 if t < k2 else T2) - 700) / 100), res)
report("random search (25k episodes)", lambda t, o: np.full(len(o), rs_best[t]), res)

curves, ppoJ = {}, []
for s in range(3):
    t0 = time.time(); ac, curve = ppo(s); curves[s] = curve
    torch.save(ac.state_dict(), os.path.join(HERE, f"85_ppo_s{s}.pt"))
    print(f"  [PPO seed {s} trained in {time.time() - t0:.0f} s]")
    ppoJ.append(report(f"PPO seed {s}", policy_fn(ac), res))
print(f"  PPO over 3 training seeds: J = {np.mean(ppoJ):.3f} +/- {np.std(ppoJ, ddof=1):.3f} (sd over seeds)")
olJ = []
for s in range(3):                                    # ablation: same PPO, but blind to the film
    ac_ol, curve = ppo(s, open_loop=True); curves[f"ol{s}"] = curve
    f = policy_fn(ac_ol)
    olJ.append(report(f"PPO open-loop seed {s}", lambda t, o, f=f: f(t, np.c_[o[:, :1], np.zeros((len(o), 4))]), res))
print(f"  PPO open-loop over 3 seeds: J = {np.mean(olJ):.3f} +/- {np.std(olJ, ddof=1):.3f}")
# oracle needs delta: build the closure per test episode
rng = np.random.default_rng(TEST_SEED); d_test = rng.uniform(-P.delta, P.delta, N_TEST)
report("oracle (knows delta)", lambda t, o: (Tact - d_test - 700) / 100, res)

# does PPO read the film as a thermometer?  set-point vs hidden delta
for s in range(3):
    Tm = res[f"PPO seed {s}"]["T"]; d = res[f"PPO seed {s}"]["d"]
    Tset_late = (Tm - d[:, None])[:, 4:].mean(1)       # actual minus delta = set-point + jitter
    print(f"  PPO seed {s}: corr(set-point over steps 5-12, delta) = {np.corrcoef(Tset_late, d)[0, 1]:+.2f}; "
          f"slope dT_set/d delta = {np.polyfit(d, Tset_late, 1)[0]:+.2f}")
np.savez(os.path.join(HERE, "85_ppo_results.npz"),
         **{f"{k}__{q}": v[q] for k, v in res.items() for q in v},
         **{f"curve_{s}": np.array(c) for s, c in curves.items()}, rs_scores=np.array(scores))
print(f"wall {time.time() - t_start:.0f} s")
