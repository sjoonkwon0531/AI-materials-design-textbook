"""Lab 8.2.1 -- composition optimization as an MDP, reward hacking, and its fixes.
Run: cd ch8/labs && python 82_comp_rl.py   (~3-4 min on one shared core)"""
import importlib, sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("82_compenv")
import lightgbm as lgb
from sklearn.model_selection import KFold
from gymnasium.utils.env_checker import check_env

t0 = time.time()
F, X, y = m.load_data()
ELEMS = ("Cu", "In", "Ga", "S", "Se"); N = 12; TARGET = 1.4; T = 8

# ---------- (1) how good is the reward model? ----------
cv = []
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
    s = m.fit_ensemble(X[tr], y[tr], M=5, seed=10 + f)
    mu, sd, _ = s.predict(X[te]); cv.append((np.abs(mu - y[te]).mean(), np.mean(np.abs(mu - y[te]) <= 1.96 * sd)))
cv = np.array(cv)
print(f"surrogate: 5-member MLP ensemble, random 5-fold MAE {cv[:,0].mean():.3f} eV, "
      f"coverage of mu+/-1.96sd {cv[:,1].mean():.2f}")
from pymatgen.core import Composition
inpal = np.array([set(e.symbol for e in Composition(f).elements) <= set(ELEMS) for f in F])
s_out = m.fit_ensemble(X[~inpal], y[~inpal], M=5, seed=99)
mu_o, sd_o, _ = s_out.predict(X[inpal])
print(f"Cu-In-Ga-S-Se compounds in data: {inpal.sum()}; held-out (palette removed from training) "
      f"MAE {np.abs(mu_o - y[inpal]).mean():.3f} eV, coverage {np.mean(np.abs(mu_o - y[inpal]) <= 1.96 * sd_o):.2f}")

sur = m.train_surrogate()                                     # all 4,601 compounds, cached
tab = m.GridTable(ELEMS, N, sur)
gbm = lgb.LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.03, min_child_samples=5,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1,
                        n_jobs=1, random_state=0).fit(X, y)
Xg = m.featurize([m.formula(ELEMS, s) for s in tab.states])
second = gbm.predict(Xg)                                      # an independent "second opinion"
St = np.array(tab.states)
tie = (St[:, 3] == St[:, 4]) & (St[:, 3] > 0)                 # S = Se: Magpie "mode" features flip
print(f"grid: {len(tab.states)} compositions (k=5, N={N}); charge-balanced {tab.balanced.sum()}")
near = np.abs(tab.mu - TARGET) < 0.05
print(f"|mu - {TARGET}| < 0.05 eV: {near.sum()} compositions, of which charge-balanced {(near & tab.balanced).sum()}")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
print(f"detectors of 'not charge balanced' over the grid (ROC-AUC): sigma {roc_auc_score(~tab.balanced, tab.sigma):.2f}, "
      f"5-NN distance {roc_auc_score(~tab.balanced, tab.dist):.2f}, |MLP-LGBM| {roc_auc_score(~tab.balanced, np.abs(tab.mu-second)):.2f}")
print(f"fraction of grid beyond d95 (Lab 6.15 AD threshold {tab.d95:.2f}): {(tab.dist > tab.d95).mean():.3f}")
print(f"balanced family: mean sigma at S=Se ties {tab.sigma[tab.balanced & tie].mean():.3f} vs others "
      f"{tab.sigma[tab.balanced & ~tie & (St[:,0]==3)].mean():.3f}")

o = np.argsort(np.abs(tab.mu - TARGET))[:8]
print("closest to target (formula: mu, sigma, LGBM, anion fraction, balanced):")
for i in o:
    print(f"   {m.formula(ELEMS, tab.states[i]):16s} {tab.mu[i]:.2f} {tab.sigma[i]:.2f} {second[i]:.2f} "
          f"{(St[i,3]+St[i,4])/N:.2f} {bool(tab.balanced[i])}")
fs = ["Cu3In3S2.9Se3.1", "Cu3In3S3Se3", "Cu3In3S3.1Se2.9"]
mu_t, sd_t, _ = sur.predict(m.featurize(fs))
print("mode-feature tie: " + ", ".join(f"{f} {a:.2f}+/-{b:.2f}" for f, a, b in zip(fs, mu_t, sd_t)))
print("balanced CuInX2/CuGaX2 end members (mu): " + ", ".join(
      f"{m.formula(ELEMS, s)} {tab.mu[tab.index[s]]:.2f}" for s in [(3,3,0,0,6),(3,3,0,6,0),(3,0,3,0,6),(3,0,3,6,0)]))

# ---------- (2) the environment ----------
env = m.CompositionEnv(tab, TARGET, T, seed=0)
check_env(env.unwrapped if hasattr(env, "unwrapped") else env, skip_render_check=True)
print("check_env: passed; obs space", env.observation_space, "| actions", env.action_space)

# ---------- (3) agents ----------
def ball_opt(S, start, T):
    """exact optimum within T edits (BFS) -- the finite-horizon DP answer for a potential-based reward"""
    frontier = {start}; seen = {start}
    for _ in range(T):
        nxt = set()
        for s in frontier:
            for a in range(env.action_space.n):
                s2 = env.next_state(s, a)
                if s2 not in seen: seen.add(s2); nxt.add(s2)
        frontier = nxt
    return max(seen, key=lambda s: S[tab.index[s]])

def run_policy(S, start, pol):
    s = start
    for t in range(T): s = env.next_state(s, pol(s))
    return s

def greedy(S):
    return lambda s: max(range(env.action_space.n), key=lambda a: S[tab.index[env.next_state(s, a)]])

def qlearn(S, episodes=20000, alpha=0.2, gamma=0.95, seed=0, eval_starts=None, every=1000):
    rng = np.random.default_rng(seed); A = env.action_space.n
    Q = np.zeros((len(tab.states), A)); curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        s = tab.states[rng.integers(len(tab.states))]
        for t in range(T):
            i = tab.index[s]
            a = rng.integers(A) if rng.random() < eps else int(np.argmax(Q[i] + 1e-9 * rng.random(A)))
            s2 = env.next_state(s, a); j = tab.index[s2]
            r = S[j] - S[i]
            Q[i, a] += alpha * (r + gamma * Q[j].max() - Q[i, a]); s = s2
        if eval_starts is not None and (ep + 1) % every == 0:
            pol = lambda s: int(np.argmax(Q[tab.index[s]]))
            curve.append(np.mean([S[tab.index[run_policy(S, s0, pol)]] for s0 in eval_starts]))
    return Q, curve

rng = np.random.default_rng(1)
starts = [tab.states[i] for i in rng.choice(len(tab.states), 200, replace=False)]
configs = {"naive": dict(), "sigma-pen (k=2)": dict(kappa=2.0), "charge-bal": dict(cb=1.0),
           "charge-bal + sigma": dict(cb=1.0, kappa=2.0)}
results, curves = {}, {}
def summarize(finals):
    idx = np.array([tab.index[s] for s in finals])
    return dict(err=np.abs(tab.mu[idx] - TARGET).mean(), bal=tab.balanced[idx].mean(), sig=tab.sigma[idx].mean(),
                disagree=np.abs(tab.mu[idx] - second[idx]).mean(), tie=tie[idx].mean(), idx=idx)
print(f"\nT = {T} edits from 200 random starts; columns: mean |mu-target| (eV), charge-balanced fraction, "
      f"mean sigma, mean |MLP-LGBM|, S=Se tie fraction")
for name, kw in configs.items():
    S = tab.score(TARGET, **kw)
    g = np.argmax(S)
    print(f"[{name}] global optimum of its score: {m.formula(ELEMS, tab.states[g])} mu={tab.mu[g]:.3f} "
          f"sigma={tab.sigma[g]:.3f} LGBM={second[g]:.2f}")
    Q, curve = qlearn(S, eval_starts=starts[:100], seed=0); curves[name] = curve
    rows = {"random policy": [run_policy(S, s0, lambda s: rng.integers(env.action_space.n)) for s0 in starts],
            "greedy (21 queries/step)": [run_policy(S, s0, greedy(S)) for s0 in starts],
            "Q-learning (20k ep.)": [run_policy(S, s0, lambda s: int(np.argmax(Q[tab.index[s]]))) for s0 in starts],
            "T-ball optimum (BFS)": [ball_opt(S, s0, T) for s0 in starts]}
    opt_sc = np.array([S[tab.index[s]] for s in rows["T-ball optimum (BFS)"]])
    for rname, finals in rows.items():
        r = summarize(finals); sc = np.array([S[tab.index[s]] for s in finals])
        results[(name, rname)] = r
        print(f"   {rname:26s} err {r['err']:.3f} | bal {r['bal']:.2f} | sigma {r['sig']:.3f} | "
              f"|MLP-LGBM| {r['disagree']:.2f} | tie {r['tie']:.2f} | reaches optimum {np.mean(np.isclose(sc, opt_sc)):.2f}")
    fin = rows["Q-learning (20k ep.)"]
    top = sorted(set(fin), key=lambda s: -fin.count(s))[:3]
    print("   most frequent Q-learning end points:", ", ".join(f"{m.formula(ELEMS, s)}({fin.count(s)})" for s in top))

np.savez(os.path.join(os.path.dirname(__file__), "82_comp_rl.npz"), mu=tab.mu, sigma=tab.sigma, second=second,
         balanced=tab.balanced, tie=tie, states=St, dist=tab.dist,
         **{f"curve_{i}": np.array(c) for i, c in enumerate(curves.values())},
         **{f"idx_{i}": results[(n, 'Q-learning (20k ep.)')]["idx"] for i, n in enumerate(configs)})
print(f"wall {time.time() - t0:.0f} s")
