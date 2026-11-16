"""Ex 8.4.2 (illustration), 8.4.3 (wider DR range, extra observations), 8.4.5 (Lagrangian constrained DR)."""
import json, os, sys, time
import numpy as np
from common import lab, LABS
E = lab("84_thermal_env")
t0 = time.time()
SC7 = np.array([1, 1, 0.1, 1, 10, 1, 1, 1.0]); SC9 = np.array([1, 1, 0.1, 1, 10, 1, 1, 1, 1, 1.0])

class Furnace9(E.FurnaceBatch):
    def obs(self):
        o, e = super().obs(); return np.hstack([o, self.ubuf[:, 2:4]]), e

class FurnaceAdv(E.FurnaceBatch):
    """adversary re-chooses h in [0.05, 0.2] every step (state-wise worst, myopic)."""
    def step(self, u):
        xs = self.t + 1
        self.h = np.where(self.x >= E.target(xs), 0.05, 0.2)
        return super().step(u)

def feats(o): return np.hstack([o, np.ones((len(o), 1))]) * (SC9 if o.shape[1] == 9 else SC7)

def plants(n, rng, dmax=4):
    return (np.exp(rng.uniform(np.log(0.05), np.log(0.2), n)), rng.uniform(0.16, 0.24, n), rng.integers(1, dmax + 1, n))

def rollout(th, h, G, d, n, rng, cls=E.FurnaceBatch):
    env = cls(n, h, G, d, rng); C = np.zeros(n); xmax = np.zeros(n)
    for t in range(E.T_EP):
        o, e = env.obs(); u = feats(o) @ th
        env.I = np.clip(env.I + e, -10, 10); env.e_prev = e
        C += env.step(u); xmax = np.maximum(xmax, env.x)
    return C / E.T_EP, xmax

def train(seed, dmax=4, cls=E.FurnaceBatch, nf=8, iters=700, n=128, lr=0.02, lag=None, d_lim=0.05, eta_l=2.0):
    rng = np.random.default_rng(seed); th = np.zeros(nf); m = np.zeros(nf); v = np.zeros(nf); lam = 0.0; Lh = []
    for it in range(iters):
        sig = 0.15 * (1 - it / iters) + 0.03
        h, G, d = plants(n, rng, dmax); env = cls(n, h, G, d, rng)
        Phi, A, C, CC = [], [], [], []; flag = np.zeros(n, bool)
        for t in range(E.T_EP):
            o, e = env.obs(); f = feats(o); mu = f @ th; a = mu + sig * rng.standard_normal(n)
            env.I = np.clip(env.I + e, -10, 10); env.e_prev = e
            C.append(env.step(a)); Phi.append(f); A.append((a - mu) / sig ** 2)
            new = (env.x > 1.05) & ~flag; CC.append(new.astype(float)); flag |= new   # cost: first crossing
        C, CC = np.array(C), np.array(CC)
        G2 = -np.cumsum(C[::-1], 0)[::-1]
        if lag is not None:
            G2 = G2 - lam * np.cumsum(CC[::-1], 0)[::-1]
            lam = max(0.0, lam + eta_l * (flag.mean() - d_lim)); Lh.append((lam, flag.mean()))
        adv = G2 - G2.mean(1, keepdims=True); adv /= adv.std() + 1e-8
        grad = np.einsum("tn,tn,tnk->k", adv, np.array(A), np.array(Phi)) / n
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
        th += lr * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
    return th, Lh

def report(name, ths, cls=E.FurnaceBatch):
    H = [0.05, 0.07, 0.1, 0.14, 0.2]; out = []
    for th in ths:
        g = np.array([[rollout(th, h, 0.2, d, 200, np.random.default_rng(1000 + 10 * i + j), cls)[0].mean()
                       for j, d in enumerate([1, 2, 3, 4, 5])] for i, h in enumerate(H)])
        c, xm = rollout(th, *plants(2000, np.random.default_rng(99)), 2000, np.random.default_rng(98), cls)
        out.append((g, c.mean(), np.quantile(c, 0.9), (xm > 1.05).mean()))
    g = 1e3 * np.array([o[0] for o in out])
    print(f"{name:<26s} nominal {g[:,2,1].mean():5.1f}+/-{g[:,2,1].std():4.1f} | worst d<=4 {g[:,:,:4].max((1,2)).mean():5.1f} | "
          f"d=5 col worst {g[:,:,4].max(1).mean():5.1f}+/-{g[:,:,4].max(1).std():4.1f} | d=5,h=0.1 {g[:,2,4].mean():5.1f} | "
          f"DR-test mean {1e3*np.mean([o[1] for o in out]):5.1f} p90 {1e3*np.mean([o[2] for o in out]):5.1f} "
          f"overshoot {100*np.mean([o[3] for o in out]):4.1f}% (seeds {', '.join('%.1f' % (100*o[3]) for o in out)}) | {time.time()-t0:.0f} s", flush=True)
    print("     d=5 column (h=0.05..0.2):", " ".join("%.1f" % x for x in g[:, :, 4].mean(0)),
          "| d=4 column:", " ".join("%.1f" % x for x in g[:, :, 3].mean(0)), flush=True)
    return out

task = sys.argv[1]
if task == "842":
    res = json.load(open(os.path.join(LABS, "84_domain_rand_results.json")))
    for reg in ("nominal", "DR"):
        th = np.array(res[reg][0]["theta"])
        fixed = {h: 1e3 * rollout(th, h, 0.2, 2, 500, np.random.default_rng(5))[0].mean() for h in (0.05, 0.07, 0.1, 0.14, 0.2)}
        adv = 1e3 * rollout(th, 0.1, 0.2, 2, 500, np.random.default_rng(5), FurnaceAdv)[0].mean()
        print(f"{reg:<8s} d=2,G=0.2: fixed h -> " + ", ".join(f"h={h:g}: {c:.1f}" for h, c in fixed.items()) +
              f" | worst fixed {max(fixed.values()):.1f} K | per-step adversarial h {adv:.1f} K")
elif task == "843":
    report("DR d in 1..5", [train(s, dmax=5)[0] for s in range(3)])
    report("DR d 1..4 + u_{t-3},u_{t-4}", [train(s, cls=Furnace9, nf=10)[0] for s in range(3)], Furnace9)
    report("DR d 1..5 + u_{t-3},u_{t-4}", [train(s, dmax=5, cls=Furnace9, nf=10)[0] for s in range(3)], Furnace9)
elif task == "845":
    ths = []
    for s in range(3):
        th, Lh = train(s, lag=True); ths.append(th); Lh = np.array(Lh)
        print(f"  seed {s}: lambda at it 100/300/700 = {Lh[99,0]:.2f}/{Lh[299,0]:.2f}/{Lh[-1,0]:.2f}; "
              f"train overshoot frac (last 100 it) {Lh[-100:,1].mean():.3f}; max lambda {Lh[:,0].max():.2f}", flush=True)
    report("DR + Lagrangian (P<=5%)", ths)
