"""Lab 8.2.4 -- two independent learners vs one centralized learner on the 'climbing game'
(a cooperative matrix game). Read as: agent 1 picks the precursor route (A/B/C), agent 2 the anneal
recipe (a/b/c); both receive the same yield-like payoff. Run: python 82_marl.py (~20 s)"""
import numpy as np, time

R = np.array([[11, -30, 0],
              [-30, 7, 6],
              [0, 0, 5]], float)           # climbing game payoff (shared reward); optimum (A, a) = 11
NOISE = 1.0                                  # Gaussian noise on each observed payoff

def eps_at(t, T): return max(0.02, 1.0 - t / (0.7 * T))

def independent(seed, T=3000, alpha=0.1):
    rng = np.random.default_rng(seed); Q1 = np.zeros(3); Q2 = np.zeros(3)
    for t in range(T):
        e = eps_at(t, T)
        a = rng.integers(3) if rng.random() < e else int(np.argmax(Q1))
        b = rng.integers(3) if rng.random() < e else int(np.argmax(Q2))
        r = R[a, b] + NOISE * rng.normal()
        Q1[a] += alpha * (r - Q1[a]); Q2[b] += alpha * (r - Q2[b])   # each treats the other as environment
    return int(np.argmax(Q1)), int(np.argmax(Q2))

def optimistic(seed, T=3000, alpha=0.1):
    """'lenient'-style variant: each agent only moves Q up quickly and down slowly (forgives partner's exploration)"""
    rng = np.random.default_rng(seed); Q1 = np.zeros(3); Q2 = np.zeros(3)
    for t in range(T):
        e = eps_at(t, T)
        a = rng.integers(3) if rng.random() < e else int(np.argmax(Q1))
        b = rng.integers(3) if rng.random() < e else int(np.argmax(Q2))
        r = R[a, b] + NOISE * rng.normal()
        for Q, k in ((Q1, a), (Q2, b)):
            d = r - Q[k]; Q[k] += (alpha if d > 0 else 0.1 * alpha) * d
    return int(np.argmax(Q1)), int(np.argmax(Q2))

def joint(seed, T=3000, alpha=0.1):
    """centralized learner over the 9 joint actions (what a single controller with both knobs would do)"""
    rng = np.random.default_rng(seed); Q = np.zeros((3, 3))
    for t in range(T):
        e = eps_at(t, T)
        if rng.random() < e: a, b = rng.integers(3), rng.integers(3)
        else: a, b = np.unravel_index(np.argmax(Q), Q.shape)
        r = R[a, b] + NOISE * rng.normal(); Q[a, b] += alpha * (r - Q[a, b])
    return tuple(int(x) for x in np.unravel_index(np.argmax(Q), Q.shape))

if __name__ == "__main__":
    t0 = time.time(); names = "ABC"; n = 200
    for label, fn in [("independent Q-learners", independent), ("independent, lenient update", optimistic),
                      ("centralized joint-action Q", joint)]:
        res = [fn(s) for s in range(n)]
        cnt = {}
        for a, b in res: cnt[(a, b)] = cnt.get((a, b), 0) + 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])
        mean_pay = np.mean([R[a, b] for a, b in res])
        print(f"{label:28s}: " + ", ".join(f"({names[a]},{names[b].lower()})={c/n:.2f}" for (a, b), c in top[:4]) +
              f" | mean converged payoff {mean_pay:.2f}")
    # why: what agent 1 'sees' for each action while agent 2 explores uniformly
    print("expected payoff of agent-1 actions vs a uniformly random partner:", np.round(R.mean(1), 2))
    print(f"wall {time.time()-t0:.0f} s")
