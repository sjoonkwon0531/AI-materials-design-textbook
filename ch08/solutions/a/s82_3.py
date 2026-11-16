"""Ex 8.2.3: drop the 22 Magpie 'mode' features from the surrogate; redo family curves and Q-learning."""
import importlib, sys, os, time
import numpy as np
sys.path.insert(0, "../ch08/labs")
m = importlib.import_module("82_compenv")
from sklearn.model_selection import KFold
import pandas as pd
t0 = time.time()
d = pd.read_parquet(os.path.join(m.DATA, "magpie_expt_gap.parquet"))
cols = list(d.columns[2:]); keep = np.array(["mode" not in c for c in cols])
print("features total", len(cols), "mode", (~keep).sum(), "kept", keep.sum())
F, X, y = m.load_data(); Xk = X[:, keep]
cv = []
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xk)):
    s = m.fit_ensemble(Xk[tr], y[tr], M=5, seed=10 + f)
    mu, sd, _ = s.predict(Xk[te]); cv.append((np.abs(mu - y[te]).mean(), np.mean(np.abs(mu - y[te]) <= 1.96 * sd)))
cv = np.array(cv); print(f"no-mode surrogate: 5-fold MAE {cv[:,0].mean():.3f} (folds {np.round(cv[:,0],3).tolist()}), coverage {cv[:,1].mean():.2f}  [{time.time()-t0:.0f}s]")
sur = m.fit_ensemble(Xk, y, M=5, seed=0)
ELEMS = ("Cu", "In", "Ga", "S", "Se"); N = 12; TARGET = 1.4; T = 8
states = m.grid_states(5, N)
Xg = m.featurize([m.formula(ELEMS, s) for s in states])[:, keep]
mu, sig, _ = sur.predict(Xg)
class Tab: pass
tab = Tab(); tab.elements = ELEMS; tab.N = N; tab.states = states; tab.index = {s: i for i, s in enumerate(states)}
tab.mu, tab.sigma = mu, sig; tab.balanced = np.array([m.charge_balanced(ELEMS, s) for s in states]); tab.dist = np.zeros(len(states)); tab.d95 = 1.0
tab.score = lambda target, kappa=0.0, lam=0.0, cb=0.0: -np.abs(tab.mu - target) - kappa * tab.sigma - cb * (~tab.balanced)
St = np.array(states); tie = (St[:, 3] == St[:, 4]) & (St[:, 3] > 0)
old = np.load("../ch08/labs/82_grid_CuInGaSSe_N12.npz")
print("family curves (x = #S of 6 anions): new mu+/-sd | old mu+/-sd")
for name, cat in [("Cu3In3", (3, 3, 0)), ("Cu3In1Ga2", (3, 1, 2)), ("Cu3Ga3", (3, 0, 3))]:
    line = []
    for x in range(7):
        i = tab.index[cat + (x, 6 - x)]
        line.append(f"{x}:{mu[i]:.2f}/{sig[i]:.2f}|{old['mu'][i]:.2f}/{old['sigma'][i]:.2f}")
    print("  ", name, "  ".join(line))
    v = np.array([mu[tab.index[cat + (x, 6 - x)]] for x in range(7)])
    vo = np.array([old['mu'][tab.index[cat + (x, 6 - x)]] for x in range(7)])
    print(f"     dip at x=3 vs mean(x=2,4): new {v[3]-0.5*(v[2]+v[4]):+.3f}, old {vo[3]-0.5*(vo[2]+vo[4]):+.3f}; monotone new {np.all(np.diff(v)>0)}")
bal = tab.balanced
print(f"balanced tie sigma {sig[bal & tie].mean():.3f} vs others {sig[bal & ~tie & (St[:,0]==3)].mean():.3f}; |mu-1.4|<0.05: {(np.abs(mu-1.4)<0.05).sum()}, balanced {((np.abs(mu-1.4)<0.05)&bal).sum()}")
print(f"grid-wide mean sigma new {sig.mean():.3f} old {old['sigma'].mean():.3f}; corr(new mu, old mu) {np.corrcoef(mu, old['mu'])[0,1]:.3f}")
env = m.CompositionEnv(tab, TARGET, T, seed=0)
def run_policy(S, s0, pol):
    s = s0
    for t in range(T): s = env.next_state(s, pol(s))
    return s
def ball_opt(S, start):
    frontier = {start}; seen = {start}
    for _ in range(T):
        nxt = set()
        for s in frontier:
            for a in range(21):
                s2 = env.next_state(s, a)
                if s2 not in seen: seen.add(s2); nxt.add(s2)
        frontier = nxt
    return max(seen, key=lambda s: S[tab.index[s]])
def qlearn(S, episodes=20000, alpha=0.2, gamma=0.95, seed=0):
    rng = np.random.default_rng(seed); A = 21; Q = np.zeros((len(states), A))
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes)); s = states[rng.integers(len(states))]
        for t in range(T):
            i = tab.index[s]
            a = rng.integers(A) if rng.random() < eps else int(np.argmax(Q[i] + 1e-9 * rng.random(A)))
            s2 = env.next_state(s, a); j = tab.index[s2]
            Q[i, a] += alpha * (S[j] - S[i] + gamma * Q[j].max() - Q[i, a]); s = s2
    return Q
rng = np.random.default_rng(1)
starts = [states[i] for i in rng.choice(len(states), 200, replace=False)]
for name, kw in {"naive": {}, "charge-bal + sigma": dict(cb=1.0, kappa=2.0)}.items():
    S = tab.score(TARGET, **kw); g = np.argmax(S)
    print(f"[{name}] global optimum {m.formula(ELEMS, states[g])} mu={mu[g]:.3f} sigma={sig[g]:.3f} bal={bal[g]} tie={tie[g]}")
    Q = qlearn(S)
    for rname, fin in {"Q-learning": [run_policy(S, s0, lambda s: int(np.argmax(Q[tab.index[s]]))) for s0 in starts],
                       "BFS optimum": [ball_opt(S, s0) for s0 in starts]}.items():
        idx = np.array([tab.index[s] for s in fin])
        print(f"   {rname:12s} err {np.abs(mu[idx]-TARGET).mean():.3f} | bal {bal[idx].mean():.2f} | sigma {sig[idx].mean():.3f} | tie {tie[idx].mean():.2f}")
    top = sorted(set(fin), key=lambda s: -fin.count(s))[:3]
    fin = [run_policy(S, s0, lambda s: int(np.argmax(Q[tab.index[s]]))) for s0 in starts]
    print("   most frequent Q end points:", ", ".join(f"{m.formula(ELEMS, s)}({fin.count(s)})" for s in sorted(set(fin), key=lambda s: -fin.count(s))[:3]))
print(f"wall {time.time()-t0:.0f} s")
