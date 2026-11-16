"""Lab 8.1 (81_bandit): 10-armed 'dopant screening' bandit, epsilon-greedy vs UCB.
Each arm = one candidate dopant; pulling = one synthesis+measurement;
reward = measured property improvement ~ N(mu_k, 1), mu_k ~ N(0, 1).
Vectorized over R independent runs (problems). Deterministic seed."""
import numpy as np, time

K, T, R = 10, 1000, 2000
rng0 = np.random.default_rng(81)
MU = rng0.normal(0.0, 1.0, size=(R, K))          # true means, one problem per run
best = MU.max(1)

def run(method, param, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((R, K)); N = np.zeros((R, K))
    rew = np.zeros(T); opt = np.zeros(T); regret = np.zeros(T)
    rows = np.arange(R); astar = MU.argmax(1)
    for t in range(T):
        if method == "eps":
            greedy = np.argmax(Q + 1e-9 * rng.random((R, K)), 1)   # random tie-break
            explore = rng.random(R) < param
            a = np.where(explore, rng.integers(0, K, R), greedy)
        else:  # UCB1-style: untried arms first, then Q + c sqrt(ln t / N)
            bonus = param * np.sqrt(np.log(t + 1) / np.maximum(N, 1e-12))
            score = np.where(N == 0, np.inf, Q + bonus)
            a = np.argmax(score + 1e-9 * rng.random((R, K)), 1)
        r = MU[rows, a] + rng.normal(0, 1, R)
        N[rows, a] += 1
        Q[rows, a] += (r - Q[rows, a]) / N[rows, a]               # sample-average update
        rew[t] = r.mean(); opt[t] = (a == astar).mean()
        regret[t] = (best - MU[rows, a]).mean()
    return rew, opt, np.cumsum(regret)

if __name__ == "__main__":
    t0 = time.time()
    cfg = [("eps", 0.0, "greedy (eps=0)"), ("eps", 0.01, "eps-greedy 0.01"),
           ("eps", 0.1, "eps-greedy 0.1"), ("ucb", 2.0, "UCB c=2")]
    out = {}
    print(f"K={K} arms, T={T} pulls, R={R} problems, mu_k~N(0,1), noise sd=1")
    print(f"{'method':18s} {'avg reward(last100)':>20s} {'%optimal(last100)':>18s} {'cum.regret@1000':>16s}")
    for m, p, name in cfg:
        rew, opt, creg = run(m, p, seed=1)
        out[name] = (rew, opt, creg)
        print(f"{name:18s} {rew[-100:].mean():20.3f} {100*opt[-100:].mean():17.1f}% {creg[-1]:16.1f}")
    print(f"mean of best arm over problems = {best.mean():.3f}")
    np.savez("../ch08/data/81_bandit.npz",
             **{k.replace(' ', '_'): np.stack(v) for k, v in out.items()})
    print(f"runtime {time.time()-t0:.1f} s")
