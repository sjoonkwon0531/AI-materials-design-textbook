"""Lab 8.4 (a): nominal vs domain-randomised vs worst-case (CVaR) policy-gradient training of a furnace
controller, tested on shifted 'real' plants.  REINFORCE with a time-dependent batch baseline (Sec.8.1.4),
Gaussian linear policy on the 7 observation features + bias, Adam.  3 training seeds per regime.
Usage: python 84_domain_rand.py    (~2-4 min, numpy only)"""
import os, time, json, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("env84", os.path.join(HERE, "84_thermal_env.py"))
E = importlib.util.module_from_spec(_s); _s.loader.exec_module(E)
t0 = time.time()
SCALE = np.array([1, 1, 0.1, 1, 10, 1, 1, 1.0])          # feature scaling (integral /10, ramp slope x10), bias


def feats(o): return np.hstack([o, np.ones((len(o), 1))]) * SCALE


def sample_plants(regime, n, rng):
    if regime == "nominal": return E.NOM["h"], E.NOM["G"], E.NOM["d"]
    h = np.exp(rng.uniform(np.log(0.05), np.log(0.2), n))           # heat-loss coefficient, x0.5 - x2
    G = rng.uniform(0.16, 0.24, n)                                    # lamp gain +/- 20 %
    d = rng.integers(1, 5, n)                                         # dead time 1-4 s
    return h, G, d


def train(regime, seed, iters=700, n=128, lr=0.02, cvar=None):
    rng = np.random.default_rng(seed)
    th = np.zeros(8); m = np.zeros(8); v = np.zeros(8); b1, b2 = 0.9, 0.999
    for it in range(iters):
        sig = 0.15 * (1 - it / iters) + 0.03
        h, G, d = sample_plants(regime, n, rng)
        env = E.FurnaceBatch(n, h, G, d, rng)
        Phi, A, C = [], [], []
        for t in range(E.T_EP):
            o, e = env.obs(); f = feats(o); mu = f @ th
            a = mu + sig * rng.standard_normal(n)
            env.I = np.clip(env.I + e, -10, 10); env.e_prev = e
            C.append(env.step(a)); Phi.append(f); A.append((a - mu) / sig ** 2)
        C = np.array(C)                                                # (T, n)
        G2 = -np.cumsum(C[::-1], 0)[::-1]                              # reward-to-go
        keep = np.ones(n, bool)
        if cvar is not None:                                           # keep the worst cvar-fraction of episodes
            keep = C.sum(0) >= np.quantile(C.sum(0), 1 - cvar)
        Gk = G2[:, keep]; adv = Gk - Gk.mean(1, keepdims=True)
        adv /= adv.std() + 1e-8
        grad = np.einsum("tn,tn,tnk->k", adv, np.array(A)[:, keep], np.array(Phi)[:, keep]) / keep.sum()
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        th += lr * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + 1e-8)   # ascent
    return th


def policy(th): return lambda o: feats(o) @ th


H_GRID = np.array([0.05, 0.07, 0.1, 0.14, 0.2]); D_GRID = np.array([1, 2, 3, 4, 5])
REGIMES = [("nominal", "nominal", None), ("DR", "dr", None), ("DR-CVaR(25%)", "dr", 0.25)]
res = {}
for name, reg, cv in REGIMES:
    grids, ths = [], []
    for seed in (0, 1, 2):
        th = train(reg, seed, cvar=cv); ths.append(th.tolist())
        g = np.zeros((len(H_GRID), len(D_GRID))); ov = np.zeros_like(g)
        for i, h in enumerate(H_GRID):
            for j, d in enumerate(D_GRID):
                r = E.rollout(policy(th), h, E.NOM["G"], d, 200, np.random.default_rng(1000 + 10 * i + j))
                g[i, j] = r["cost"].mean(); ov[i, j] = (r["xmax"] > 1.05).mean()
        grids.append(g)
        rr = E.rollout(policy(th), *sample_plants("dr", 2000, np.random.default_rng(99)), 2000, np.random.default_rng(98))
        res.setdefault(name, []).append(dict(theta=th.tolist(), grid=g.tolist(), over=ov.tolist(),
                                             dr_mean=rr["cost"].mean(), dr_p90=np.quantile(rr["cost"], 0.9),
                                             dr_cvar25=np.sort(rr["cost"])[-500:].mean(),
                                             dr_over=(rr["xmax"] > 1.05).mean()))
    G = np.array(grids)
    i0, j0 = 2, 1                                                      # nominal: h = 0.1, d = 2
    print("%-13s nominal plant %5.1f +/- %4.1f K | h=0.05,d=4 %5.1f | h=0.2,d=1 %5.1f | worst of grid(d<=4) %5.1f +/- %4.1f K | "
          "grid d=5 worst %6.1f | DR-test mean %5.1f, p90 %5.1f, CVaR25 %5.1f K, overshoot>50K %4.1f%%" % (
          name, 1e3 * G[:, i0, j0].mean(), 1e3 * G[:, i0, j0].std(), 1e3 * G[:, 0, 3].mean(), 1e3 * G[:, 4, 0].mean(),
          1e3 * G[:, :, :4].max((1, 2)).mean(), 1e3 * G[:, :, :4].max((1, 2)).std(), 1e3 * G[:, :, 4].max(1).mean(),
          1e3 * np.mean([r["dr_mean"] for r in res[name]]), 1e3 * np.mean([r["dr_p90"] for r in res[name]]), 1e3 * np.mean([r["dr_cvar25"] for r in res[name]]),
          100 * np.mean([r["dr_over"] for r in res[name]])), flush=True)
    print("   mean grid (K), rows h=" + ",".join(f"{h:g}" for h in H_GRID) + "; cols d=1..5")
    for i in range(len(H_GRID)): print("     " + " ".join("%6.1f" % (1e3 * G[:, i, j].mean()) for j in range(len(D_GRID))))
    print("   theta (seed 0): " + " ".join("%.3f" % t for t in res[name][0]["theta"]))
json.dump(res, open(os.path.join(HERE, "84_domain_rand_results.json"), "w"))
print("wall %.0f s" % (time.time() - t0))
