"""Ex 8.2.6: starvation penalty (beta * age of oldest lot at window end) and FIFO warm start for the REINFORCE dispatcher."""
import sys, importlib, time, numpy as np
sys.path.insert(0, "../../../ch08/labs")
M = importlib.import_module("82_fab")
Fab, RAW, NF = M.Fab, M.RAW, M.NF
# feature standardization exactly as in the lab (FIFO run, seed 777)
fab = Fab(0.2875, 300, 777); FF = []
while True:
    s = fab.pending_decision()
    if s is None: break
    F = fab.features(s); FF.append(F); fab._start(s, int(np.argmax(F[:, 4])))
FF = np.vstack(FF); M.MU = FF.mean(0); M.SD = FF.std(0) + 1e-8
std = M.standardize

def oldest_age(fab, tq):
    arr = np.array([l["arr"] for l in fab.lots]); out = np.array([l.get("out", np.inf) for l in fab.lots])
    inside = (arr[None, :] <= tq[:, None]) & (out[None, :] > tq[:, None])
    a = np.where(inside, arr[None, :], np.inf).min(1)
    return np.where(np.isfinite(a), tq - a, 0.0)

def train(beta, seed, warm=0.0, iters=120, K=8, lr=0.3, horizon=500.0, rate=0.2875, window=40.0):
    rng = np.random.default_rng(seed); theta = np.zeros(NF); H = warm + horizon; hist = []
    for it in range(iters):
        scen = int(rng.integers(10**6)); runs = []
        for k in range(K):
            fab = Fab(rate, H, scen); prng = np.random.default_rng(scen * 31 + k); dec = []
            while True:
                s = fab.pending_decision()
                if s is None: break
                F = fab.features(s)
                if fab.t < warm: fab._start(s, int(np.argmax(F[:, 4]))); continue      # FIFO warm-up, not learned
                Phi = std(F); z = Phi @ theta; p = np.exp(z - z.max()); p /= p.sum()
                a = prng.choice(len(p), p=p); dec.append((fab.t, Phi[a] - p @ Phi)); fab._start(s, a)
            T_, A_ = np.array(fab.log).T; runs.append((T_, A_, dec, fab))
        def G(T_, A_, f, t):
            te = np.minimum(t + window, H)
            g = -(np.interp(te, T_, A_) - np.interp(t, T_, A_)) / RAW
            return g - beta * oldest_age(f, te) if beta > 0 else g
        grad = np.zeros(NF); nd = 0
        for T_, A_, dec, f in runs:
            if not dec: continue
            t = np.array([d[0] for d in dec]); g = G(T_, A_, f, t)
            b = np.mean([G(T2, A2, f2, t) for T2, A2, _, f2 in runs], axis=0)
            grad += ((g - b)[:, None] * np.array([d[1] for d in dec])).sum(0); nd += len(dec)
        theta += lr * grad / max(nd, 1)
    return theta

def test(theta, rate, n=30):
    pol = lambda F: int(np.argmax(std(F) @ theta))
    R = [M.simulate(pol, rate, 2000.0, 5000 + s, warm=200.0) for s in range(n)]
    return {k: np.array([r[k] for r in R]) for k in ("wip", "littleCT", "ct", "late", "oldest", "stuck")}

if __name__ == "__main__":
    t0 = time.time(); names = ["step", "p_now", "remaining", "slack", "wait_q", "time_in_fab", "next_queue"]
    ref = {}
    for nm, rule in [("least slack", M.RULES["least slack"]), ("LBFS", M.RULES["LBFS"])]:
        R = [M.simulate(rule, 0.2875, 2000.0, 5000 + s, warm=200.0) for s in range(30)]
        ref[nm] = np.array([r["wip"] for r in R])
        print(f"{nm:12s} WIP {ref[nm].mean():.2f} | LittleCT {np.mean([r['littleCT'] for r in R]):.1f} | CT(fin) {np.mean([r['ct'] for r in R]):.1f} | late {np.mean([r['late'] for r in R]):.2f} | oldest {np.mean([r['oldest'] for r in R]):.0f} h")
    configs = [(b, 0.0) for b in (0.0, 0.05, 0.2)] + [(b, 200.0) for b in (0.0, 0.2)]
    for beta, warm in configs:
        for sd in range(3):
            tt = time.time(); th = train(beta, sd, warm=warm); res = test(th, 0.2875)
            d = res["wip"] - ref["least slack"]
            print(f"beta {beta:4.2f} warm {warm:3.0f} seed {sd}: theta " + " ".join(f"{n}{t:+.2f}" for n, t in zip(names, th)) +
                  f" | WIP {res['wip'].mean():.2f} (vs LS {d.mean():+.2f}+/-{d.std()/np.sqrt(30):.2f}) | LittleCT {res['littleCT'].mean():.1f} | CT(fin) {res['ct'].mean():.1f} | late {res['late'].mean():.2f} | oldest {res['oldest'].mean():.0f} h ({time.time()-tt:.0f} s)")
            sys.stdout.flush()
    print(f"wall {time.time()-t0:.0f} s")
