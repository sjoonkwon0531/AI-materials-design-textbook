"""Ex 8.2.5: FIFO(fab) = LBFS on the 1-machine line; does it break with parallel machines?"""
import sys, importlib, heapq, time, numpy as np
sys.path.insert(0, "../ch08/labs")
M = importlib.import_module("82_fab")
ROUTE, RAW = M.ROUTE, M.RAW

class FabK(M.Fab):
    """same simulator, but station s has n_mach[s] identical machines"""
    def __init__(self, rate, horizon, seed, n_mach=(1, 1, 1)):
        super().__init__(rate, horizon, seed); self.nm = n_mach; self.busy = [[None] * k for k in n_mach]
    def idle(self, s): return any(b is None for b in self.busy[s])
    def pending_decision(self):
        while True:
            for s in range(3):
                while self.idle(s) and len(self.q[s]) == 1: self._start(s, 0)
            for s in range(3):
                if self.idle(s) and len(self.q[s]) >= 2: return s
            if not self.ev or self.ev[0][0] > self.H:
                self._advance(self.H); return None
            t, _, kind, x = heapq.heappop(self.ev); self._advance(t)
            if kind == "arr":
                lot = dict(id=len(self.lots), arr=t, due=t + self.df * RAW, step=0, qarr=t)
                self.lots.append(lot); self.wip += 1; self._enqueue(lot)
                self._push(t + self.rng.exponential(1 / self.rate), "arr", None)
            else:
                s, j = x; lot = self.busy[s][j]; self.busy[s][j] = None; lot["step"] += 1
                if lot["step"] == len(ROUTE): lot["out"] = t; self.done.append(lot); self.wip -= 1
                else: self._enqueue(lot)
    def _start(self, s, k):
        lot = self.q[s].pop(k); j = self.busy[s].index(None); self.busy[s][j] = lot
        self._push(self.t + self._ptime(lot), "done", (s, j))

def simulate(rule, rate, horizon, seed, n_mach, warm=200.0):
    fab = FabK(rate, horizon, seed, n_mach); n_dec = 0; n_dis = 0
    while True:
        s = fab.pending_decision()
        if s is None: break
        F = fab.features(s)
        a_fab, a_lb = int(np.argmax(F[:, 5])), int(np.argmax(F[:, 0]))
        n_dec += 1; n_dis += (F[a_fab, 0] != F[a_lb, 0])          # oldest-in-fab lot is NOT at the most advanced step
        fab._start(s, a_fab if rule == "FIFO (fab)" else a_lb if rule == "LBFS" else int(M.RULES[rule](F)))
    done = [l for l in fab.done if l["arr"] >= warm]
    # overtaking among finished lots: pairs (i<j by arrival) with out_j < out_i
    outs = np.array([l["out"] for l in sorted(fab.done, key=lambda l: l["arr"])])
    overt = int(np.sum(outs[1:] < np.maximum.accumulate(outs)[:-1]))
    stuck = [horizon - l["arr"] for l in fab.lots if "out" not in l]
    return dict(wip=fab.area / horizon, ct=np.mean([l["out"] - l["arr"] for l in done]), thr=len(fab.done) / horizon,
                dis=n_dis / max(n_dec, 1), overt=overt, oldest=max(stuck) if stuck else 0.0)

if __name__ == "__main__":
    t0 = time.time()
    for label, nm, rate in [("1 machine each, rate 0.2875", (1, 1, 1), 0.2875),
                            ("2 litho only, rate 0.55 (as stated)", (1, 2, 1), 0.55),
                            ("2 litho only, rate 0.35", (1, 2, 1), 0.35),
                            ("2 machines each, rate 0.55", (2, 2, 2), 0.55)]:
        util = [rate * sum(p for s, p in ROUTE if s == k) / nm[k] for k in range(3)]
        print(f"[{label}] utilization dep/litho/etch = {util[0]:.2f}/{util[1]:.2f}/{util[2]:.2f}")
        res = {}
        for rule in ["FIFO (fab)", "LBFS", "FIFO", "least slack"]:
            R = [simulate(rule, rate, 2000.0, 5000 + s, nm) for s in range(30)]
            w = np.array([r["wip"] for r in R]); res[rule] = w
            print(f"   {rule:12s} WIP {w.mean():6.2f}+/-{w.std()/np.sqrt(30):.2f} | CT {np.mean([r['ct'] for r in R]):6.1f} h | thr {np.mean([r['thr'] for r in R]):.4f} | "
                  f"decisions where oldest-in-fab != most-advanced {np.mean([r['dis'] for r in R]):.3f} | overtakes {np.mean([r['overt'] for r in R]):.1f} | oldest {np.mean([r['oldest'] for r in R]):.0f} h")
        d = res["FIFO (fab)"] - res["LBFS"]
        print(f"   paired WIP FIFO(fab) - LBFS: {d.mean():+.3f} +/- {d.std()/np.sqrt(30):.3f}; identical in {np.mean(np.abs(d)<1e-12):.2f} of scenarios  [{time.time()-t0:.0f} s]")
        sys.stdout.flush()
    # (c) Little's law from the Lab 8.7 output table (rate 0.2875 and 0.30)
    tab = {0.2875: [("FIFO",9.1,31.4,32.5,0.2831,27),("FIFO (fab)",8.7,29.2,30.5,0.2884,27),("SPT",8.9,30.5,31.6,0.2830,31),
                    ("LBFS",8.7,29.2,30.5,0.2884,27),("FBFS",17.7,61.4,61.4,0.2734,61),("least slack",7.3,25.4,26.2,0.2816,22),
                    ("random",8.8,30.6,31.5,0.2778,47),("RL seed0 (greedy)",8.1,27.5,24.9,0.2874,158),("RL seed0 (stoch)",10.4,34.4,28.1,0.2882,358),
                    ("RL seed2 (greedy)",8.2,27.8,24.8,0.2878,212),("RL from LS (greedy)",8.5,28.6,25.8,0.2885,291)],
           0.30: [("FIFO",15.2,49.5,52.2,0.2931,58),("FIFO (fab)",9.9,32.8,34.5,0.2937,35),("SPT",15.6,50.6,53.9,0.2929,60),
                  ("FBFS",32.4,105.7,114.4,0.2796,126),("least slack",10.8,35.3,36.8,0.2949,38),("random",18.4,59.2,62.6,0.2975,112),
                  ("RL seed0 (greedy)",10.2,33.5,27.9,0.2939,384),("RL seed2 (greedy)",10.0,32.9,27.6,0.2940,347),("RL from LS (greedy)",11.0,35.9,28.5,0.2946,558)]}
    from scipy.stats import spearmanr
    for rate, rows in tab.items():
        print(f"(c) rate {rate}: name  L/thr  LittleCT  CT(fin)  CT(fin)-LittleCT  oldest")
        x = []; o = []
        for n, L, lct, ct, thr, old in rows:
            W = L / thr; x.append(ct - lct); o.append(old)
            print(f"   {n:20s} {W:6.1f} {lct:6.1f} {ct:6.1f} {ct-lct:+6.1f} {old:5d}")
        r = spearmanr(x, o); print(f"   Spearman(CT(fin)-LittleCT, oldest) = {r.correlation:+.2f} (p={r.pvalue:.3f}), Pearson {np.corrcoef(x,o)[0,1]:+.2f}")
    print(f"wall {time.time()-t0:.0f} s")
