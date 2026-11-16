"""Lab 8.2.3 -- a re-entrant flow line (3 layers x deposition/litho/etch) as a Gymnasium env;
dispatching rules vs a REINFORCE-trained linear-softmax dispatcher.
Run: python 82_fab.py   (~6-8 min on one shared core)"""
import heapq, time, os, json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ST = ["dep", "litho", "etch"]
ROUTE = [(0, 0.9), (1, 1.0), (2, 0.6),        # layer 1: (station, mean processing time)
         (0, 0.6), (1, 1.5), (2, 1.0),        # layer 2
         (0, 1.1), (1, 0.7), (2, 0.8)]        # layer 3
RAW = sum(p for _, p in ROUTE)                # raw process time per lot = 8.2
LITHO = sum(p for s, p in ROUTE if s == 1)    # 3.2 litho hours per lot -> bottleneck
KMAX = 24                                     # observation pads the queue to KMAX lots
NF = 7                                        # features per lot

class Fab:
    """Discrete-event simulator, one machine per station, lognormal processing noise (CV 0.25)."""
    def __init__(self, rate, horizon, seed, due_factor=2.5):
        self.rng = np.random.default_rng(seed); self.rate, self.H, self.df = rate, horizon, due_factor
        self.t = 0.0; self.ev = []; self.q = [[] for _ in ST]; self.busy = [None] * 3
        self.lots = []; self.done = []; self.area = 0.0; self.last = 0.0; self.wip = 0; self.log = [(0.0, 0.0)]
        self._push(self.rng.exponential(1 / rate), "arr", None)
    def _push(self, t, kind, x):
        self.cnt = getattr(self, "cnt", 0) + 1; heapq.heappush(self.ev, (t, self.cnt, kind, x))
    def _advance(self, t):
        self.area += self.wip * (t - self.last); self.last = t; self.t = t
        self.log.append((t, self.area))
    def _ptime(self, lot):
        s, p = ROUTE[lot["step"]]; return p * self.rng.lognormal(-0.5 * 0.0606, 0.246)   # mean p, CV ~0.25
    def _enqueue(self, lot):
        lot["qarr"] = self.t; self.q[ROUTE[lot["step"]][0]].append(lot)
    def pending_decision(self):
        """advance until some idle station has >= 2 waiting lots (a real choice); single lots start at once"""
        while True:
            for s in range(3):
                if self.busy[s] is None and len(self.q[s]) == 1:
                    self._start(s, 0)
            for s in range(3):
                if self.busy[s] is None and len(self.q[s]) >= 2:
                    return s
            if not self.ev or self.ev[0][0] > self.H:
                self._advance(self.H); return None
            t, _, kind, x = heapq.heappop(self.ev); self._advance(t)
            if kind == "arr":
                lot = dict(id=len(self.lots), arr=t, due=t + self.df * RAW, step=0, qarr=t)
                self.lots.append(lot); self.wip += 1; self._enqueue(lot)
                self._push(t + self.rng.exponential(1 / self.rate), "arr", None)
            else:                                     # completion at station x
                lot = self.busy[x]; self.busy[x] = None; lot["step"] += 1
                if lot["step"] == len(ROUTE):
                    lot["out"] = t; self.done.append(lot); self.wip -= 1
                else:
                    self._enqueue(lot)
    def _start(self, s, k):
        lot = self.q[s].pop(k); self.busy[s] = lot
        self._push(self.t + self._ptime(lot), "done", s)
    def features(self, s):
        """per-lot features at station s: step, this step's time, remaining work after it, slack,
        time waited in this queue, time in fab, length of the queue at the lot's next station"""
        F = []
        for lot in self.q[s]:
            k = lot["step"]; p = ROUTE[k][1]; rem = sum(pp for _, pp in ROUTE[k + 1:])
            nxt = len(self.q[ROUTE[k + 1][0]]) if k + 1 < len(ROUTE) else 0
            F.append([k / 8, p, rem / RAW, (lot["due"] - self.t - p - rem) / RAW,
                      (self.t - lot["qarr"]) / RAW, (self.t - lot["arr"]) / RAW, nxt / 5])
        return np.array(F)

RULES = {
    "FIFO":        lambda F: np.argmax(F[:, 4]),                 # longest wait in this queue
    "FIFO (fab)":  lambda F: np.argmax(F[:, 5]),                 # earliest fab arrival
    "SPT":         lambda F: np.argmin(F[:, 1]),                 # shortest processing time
    "LBFS":        lambda F: np.argmax(F[:, 0]),                 # last buffer first served (most advanced)
    "FBFS":        lambda F: np.argmin(F[:, 0]),                 # first buffer first served
    "least slack": lambda F: np.argmin(F[:, 3]),
    "random":      None,
}

def simulate(policy, rate, horizon, seed, warm=0.0):
    fab = Fab(rate, horizon, seed); rng = np.random.default_rng(seed + 12345)
    while True:
        s = fab.pending_decision()
        if s is None: break
        F = fab.features(s)
        k = rng.integers(len(F)) if policy is None else int(policy(F))
        fab._start(s, k)
    done = [l for l in fab.done if l["arr"] >= warm]
    ct = np.array([l["out"] - l["arr"] for l in done])
    late = np.array([l["out"] > l["due"] for l in done])
    stuck = [horizon - l["arr"] for l in fab.lots if "out" not in l]          # still in the fab at the end
    return dict(ct=ct.mean(), ct95=np.percentile(ct, 95), thr=len(fab.done) / horizon, wip=fab.area / horizon,
                late=late.mean(), n=len(done), stuck=len(stuck), oldest=max(stuck) if stuck else 0.0,
                littleCT=fab.area / horizon / (len(fab.lots) / horizon))

class FabEnv(gym.Env):
    """Gymnasium wrapper: each step is one dispatching decision at an idle station with >= 2 waiting lots.
    obs: (KMAX, NF) feature matrix, zero-padded, plus a mask; action: index of the lot to start
    (an index beyond the queue length is taken modulo the queue length).
    reward: -(integral of WIP since the previous decision) / RAW, so the return is -(integrated WIP)/RAW and,
    by Little's law, maximizing it minimizes the mean cycle time at a given throughput."""
    metadata = {"render_modes": []}
    def __init__(self, rate=0.2875, horizon=300.0):
        super().__init__(); self.rate, self.H = rate, horizon
        self.observation_space = spaces.Dict({"lots": spaces.Box(-np.inf, np.inf, (KMAX, NF), np.float64),
                                              "mask": spaces.MultiBinary(KMAX)})
        self.action_space = spaces.Discrete(KMAX)
    def _obs(self):
        F = self.fab.features(self.s)[:KMAX]; L = np.zeros((KMAX, NF)); L[:len(F)] = F
        m = np.zeros(KMAX, np.int8); m[:len(F)] = 1
        return {"lots": L, "mask": m}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fab = Fab(self.rate, self.H, int(self.np_random.integers(2**31)))
        self.s = self.fab.pending_decision()
        while self.s is None:                     # (practically never) no decision in the horizon
            self.fab = Fab(self.rate, self.H, int(self.np_random.integers(2**31))); self.s = self.fab.pending_decision()
        self.area0 = self.fab.area
        return self._obs(), {"station": self.s}
    def step(self, a):
        self.fab._start(self.s, int(a) % min(len(self.fab.q[self.s]), KMAX))
        self.s = self.fab.pending_decision()
        r = -(self.fab.area - self.area0) / RAW; self.area0 = self.fab.area
        done = self.s is None
        return (self._obs() if not done else {"lots": np.zeros((KMAX, NF)), "mask": np.zeros(KMAX, np.int8)},
                r, done, False, {"station": self.s})

def standardize(F):
    return (F - MU) / SD

def reinforce_window(iters=200, K=8, lr=0.1, horizon=500.0, rate=0.2875, window=40.0, seed=0, theta0=None):
    """REINFORCE with time-local credit: the 'return' of a decision at time t is -(WIP integrated over
    [t, t+window]); the baseline is the same window integral averaged over the K rollouts that share the
    scenario (common random numbers). Truncating the window trades bias for variance (cf. GAE, sec 8.1)."""
    rng = np.random.default_rng(seed); theta = np.zeros(NF) if theta0 is None else np.array(theta0, float)
    hist = []
    for it in range(iters):
        scen = int(rng.integers(10**6)); runs = []
        for k in range(K):
            fab = Fab(rate, horizon, scen); prng = np.random.default_rng(scen * 31 + k); dec = []
            while True:
                s = fab.pending_decision()
                if s is None: break
                Phi = standardize(fab.features(s)); z = Phi @ theta; p = np.exp(z - z.max()); p /= p.sum()
                a = prng.choice(len(p), p=p); dec.append((fab.t, Phi[a] - p @ Phi)); fab._start(s, a)
            T_, A_ = np.array(fab.log).T
            runs.append((T_, A_, dec, fab.area / horizon))
        Gfun = lambda T_, A_, t: -(np.interp(np.minimum(t + window, horizon), T_, A_) - np.interp(t, T_, A_)) / RAW
        grad = np.zeros(NF); nd = 0
        for T_, A_, dec, _ in runs:
            if not dec: continue
            t = np.array([d[0] for d in dec]); G = Gfun(T_, A_, t)
            b = np.mean([Gfun(T2, A2, t) for T2, A2, _, _ in runs], axis=0)
            grad += ((G - b)[:, None] * np.array([d[1] for d in dec])).sum(0); nd += len(dec)
        theta += lr * grad / max(nd, 1)
        hist.append(-np.mean([r[3] for r in runs]))
    return theta, np.array(hist)

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    t0 = time.time()
    env = FabEnv(); check_env(env, skip_render_check=True)
    o, _ = env.reset(seed=0); R = 0; n = 0
    while True:
        o, r, d, tr, _ = env.step(0); R += r; n += 1
        if d: break
    print(f"check_env passed; one FabEnv episode (always first lot): {n} decisions, return {R:.2f}")
    rate = 0.2875
    print(f"route: {len(ROUTE)} steps, raw process time {RAW:.1f} h; litho load {LITHO:.1f} h/lot; "
          f"arrival rate {rate} lots/h -> litho utilization {rate*LITHO:.2f}")
    # feature standardization from a FIFO run
    fab = Fab(rate, 300, 777); FF = []
    while True:
        s = fab.pending_decision()
        if s is None: break
        F = fab.features(s); FF.append(F); fab._start(s, int(np.argmax(F[:, 4])))
    FF = np.vstack(FF); MU = FF.mean(0); SD = FF.std(0) + 1e-8
    names = ["step", "p_now", "remaining", "slack", "wait_q", "time_in_fab", "next_queue"]
    pols = dict(RULES); hists = {}
    def stoch(theta, seed=0):
        rng = np.random.default_rng(seed)
        def p(F):
            z = standardize(F) @ theta; q = np.exp(z - z.max()); q /= q.sum(); return rng.choice(len(q), p=q)
        return p
    runs = [("RL seed0", None, 0), ("RL seed1", None, 1), ("RL seed2", None, 2)]
    th0 = np.zeros(NF); th0[3] = -1.0                      # = least slack, as a softmax with temperature 1
    runs.append(("RL from least-slack", th0, 0))
    for name, init, sd in runs:
        tt = time.time()
        theta, hist = reinforce_window(iters=120, lr=0.3, seed=sd, theta0=init); hists[name] = hist
        print(f"{name}: theta = " + ", ".join(f"{n} {t:+.2f}" for n, t in zip(names, theta)) +
              f" | training mean WIP (lots, 500 h from empty) first10 {-hist[:10].mean():.2f} last10 {-hist[-10:].mean():.2f} ({time.time()-tt:.0f} s)")
        pols[name + " (greedy)"] = (lambda th: (lambda F: int(np.argmax(standardize(F) @ th))))(theta.copy())
        if name == "RL seed0": pols[name + " (stochastic)"] = stoch(theta.copy())
    pols["least slack (softmax T=1)"] = stoch(th0)
    out = {}
    print(f"\ntest: 30 scenarios, horizon 2000 h, statistics over lots released after 200 h warm-up (mean +/- SE)")
    for H_rate in [rate, 0.30]:
        print(f"arrival rate {H_rate} (litho utilization {H_rate*LITHO:.2f}):")
        for name, pol in pols.items():
            R = [simulate(pol, H_rate, 2000.0, 5000 + s, warm=200.0) for s in range(30)]
            ct = np.array([r["ct"] for r in R]); out[(H_rate, name)] = ct
            w = np.array([r["wip"] for r in R]); out[(H_rate, name, "wip")] = w
            print(f"  {name:28s} WIP {w.mean():5.1f}+/-{w.std()/np.sqrt(len(w)):.1f} | Little CT {np.mean([r['littleCT'] for r in R]):6.1f} h | "
                  f"CT(finished) {ct.mean():6.1f} | p95 {np.mean([r['ct95'] for r in R]):6.1f} | late {np.mean([r['late'] for r in R]):.2f} | "
                  f"thr {np.mean([r['thr'] for r in R]):.4f} | left in fab {np.mean([r['stuck'] for r in R]):4.1f}, oldest {np.mean([r['oldest'] for r in R]):5.0f} h")
        best_rule = min([k for k in RULES if k != "random"], key=lambda k: out[(H_rate, k, "wip")].mean())
        for name in [k for k in pols if k.startswith("RL")]:
            d = out[(H_rate, name, "wip")] - out[(H_rate, best_rule, "wip")]
            print(f"  paired WIP difference {name} - {best_rule}: {d.mean():+.2f} +/- {d.std()/np.sqrt(len(d)):.2f}")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "82_fab.npz"),
             **{"hist_" + k: v for k, v in hists.items()},
             **{"_".join(map(str, key)): v for key, v in out.items()})
    print(f"wall {time.time()-t0:.0f} s")
