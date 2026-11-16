"""Lab 81_cliff_triad.
(a) Cliff walking (Sutton & Barto Example 6.6 layout, 4x12): SARSA (on-policy) vs Q-learning (off-policy).
    Read as a 'process window': the fastest recipe runs along the failure edge (the cliff).
(b) Baird's counterexample (Sutton & Barto §11.2 layout): the deadly triad
    (function approximation + bootstrapping + off-policy weighting) makes expected semi-gradient
    TD diverge; removing any one leg restores convergence.
Runtime ~30 s. Deterministic seeds."""
import numpy as np, time

# ------------------------------------------------------------------ (a) cliff walking
H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left

def env_step(s, a):
    r, c = s; dr, dc = MOVES[a]
    r2, c2 = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
    if r2 == 3 and 1 <= c2 <= 10:                    # fell off the cliff
        return START, -100.0, False
    return (r2, c2), -1.0, (r2, c2) == GOAL

def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps: return int(rng.integers(4))
    q = Q[s]; return int(rng.choice(np.flatnonzero(q == q.max())))

def run(method, n_ep=500, eps=0.1, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((H, W, 4)); rets = np.zeros(n_ep)
    for ep in range(n_ep):
        s = START; a = eps_greedy(Q, s, eps, rng); G = 0.0
        for t in range(10_000):
            s2, r, done = env_step(s, a); G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            if method == "sarsa": target = r + (0 if done else Q[s2][a2])     # next action actually taken
            else:                 target = r + (0 if done else Q[s2].max())   # greedy next action
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = s2, a2
            if done: break
        rets[ep] = G
    return Q, rets

def greedy_path(Q):
    s = START; path = [s]
    for _ in range(100):
        s, r, done = env_step(s, int(np.argmax(Q[s]))); path.append(s)
        if done: return path
    return None

# ------------------------------------------------------------------ (b) Baird's counterexample
def baird(mode, sweeps=1000, alpha=0.01, gamma=0.99):
    """Expected (DP-style) semi-gradient update with uniform state weighting.
    Target policy pi: always 'solid' -> state 7, reward 0, so v_pi = 0 everywhere."""
    X = np.zeros((7, 8))
    for i in range(6): X[i, i] = 2; X[i, 7] = 1
    X[6, 6] = 1; X[6, 7] = 2
    if mode == "tabular": X = np.eye(7)               # remove function approximation
    w = np.ones(X.shape[1]);
    if mode != "tabular": w[6] = 10.0
    norms = []; vmax = []
    for k in range(sweeps):
        v = X @ w
        if mode == "mc": target = np.zeros(7)          # remove bootstrapping: true return is 0
        else:            target = gamma * v[6] * np.ones(7)  # E_pi[R + gamma v(s')] = gamma v(7)
        w = w + alpha / 7 * X.T @ (target - v)
        norms.append(np.linalg.norm(w)); vmax.append(np.abs(X @ w).max())
    return np.array(norms), np.array(vmax)

if __name__ == "__main__":
    t0 = time.time()
    print("(a) cliff walking, eps=0.1, alpha=0.5, gamma=1, 500 episodes, 50 runs")
    curves = {}
    for m in ("sarsa", "qlearn"):
        R = []; plens = []
        for sd in range(50):
            Q, rets = run(m, seed=sd); R.append(rets)
            p = greedy_path(Q); plens.append(len(p) - 1 if p else -1)   # -1: greedy policy loops
        R = np.array(R); curves[m] = R
        plens = np.array(plens)
        print(f"  {m:7s}: online return (episodes 401-500) = {R[:,400:].mean():7.2f} "
              f"+- {R[:,400:].mean(1).std(ddof=1)/np.sqrt(50):.2f};  greedy path length: "
              f"{np.bincount(plens[plens>0]).nonzero()[0].tolist()} "
              f"(counts {np.bincount(plens[plens>0])[np.bincount(plens[plens>0]).nonzero()[0]].tolist()}), "
              f"greedy policy loops in {(plens<0).sum()} runs")
    Qs, _ = run("sarsa", seed=0); Qq, _ = run("qlearn", seed=0)
    for name, Q in (("sarsa", Qs), ("qlearn", Qq)):
        rows = [p[0] for p in greedy_path(Q)]
        print(f"  seed-0 greedy path of {name:7s}: rows visited {sorted(set(rows))}, length {len(rows)-1}")
    print("\n(b) Baird's counterexample, expected semi-gradient updates, alpha=0.01, gamma=0.99")
    out = {}
    for mode in ("triad", "mc", "tabular"):
        n, vm = baird(mode, sweeps=5000); out[mode] = n; out[mode + "_vmax"] = vm
        print(f"  {mode:8s}: ||w|| at sweep 1/1000/5000 = {n[0]:.3g} / {n[999]:.3g} / {n[-1]:.3g};"
              f"  max|v - v_pi| = {vm[0]:.3g} / {vm[999]:.3g} / {vm[-1]:.3g}")
    np.savez("../../ch08/data/81_cliff_triad.npz", sarsa=curves["sarsa"],
             qlearn=curves["qlearn"], **{f"baird_{k}": v for k, v in out.items()})
    print(f"runtime {time.time()-t0:.1f} s")
