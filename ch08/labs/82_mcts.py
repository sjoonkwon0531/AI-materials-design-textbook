"""Lab 8.2.2 -- retrosynthesis as tree search on toy reaction networks: greedy vs MCTS (UCT) vs exact DP.
Run: python 82_mcts.py   (~1-2 min)"""
import numpy as np, math, time, json, os

LEVELS = [30, 25, 25, 20, 15, 10, 1]          # molecules per complexity level; level 0 = possible building blocks

def make_network(seed):
    rng = np.random.default_rng(seed)
    ids, lev = [], []
    for L, n in enumerate(LEVELS):
        for _ in range(n): ids.append(len(ids)); lev.append(L)
    lev = np.array(lev)
    buyable = np.zeros(len(ids), bool); buyable[lev == 0] = rng.random((lev == 0).sum()) < 0.7
    rxns = {}                                    # product -> list of (yield, [precursors])
    for mol in ids:
        L = lev[mol]
        if L == 0: continue
        if L <= 2 and rng.random() < 0.15: rxns[mol] = []; continue      # no known disconnection
        out = []
        for _ in range(rng.integers(1, 5)):
            npre = 1 if rng.random() < 0.4 else 2
            pre = []
            for _ in range(npre):
                Lp = max(0, L - rng.integers(1, 3))
                pre.append(int(rng.choice(np.where(lev == Lp)[0])))
            out.append((float(rng.uniform(0.2, 0.95)), pre))
        rxns[mol] = out
    return dict(lev=lev, buyable=buyable, rxns=Logged(rxns), target=len(ids) - 1)

class Logged(dict):
    """reaction table that records which molecules were queried (= calls to a one-step retrosynthesis model,
    cached per molecule)"""
    def __init__(self, *a):
        super().__init__(*a); self.seen = set()
    def get(self, k, default=None):
        self.seen.add(k); return super().get(k, default)
    def __getitem__(self, k):
        self.seen.add(k); return super().__getitem__(k)

def dp_best(net):
    """Exact optimum of the AND-OR tree: V(m) = 1 if buyable, else max_r y_r * prod V(p).  (Bellman for AND-OR)"""
    memo = {}
    def V(m):
        if m in memo: return memo[m]
        if net["buyable"][m]: memo[m] = 1.0; return 1.0
        best = 0.0
        for y, pre in net["rxns"].get(m, []):
            best = max(best, y * math.prod(V(p) for p in pre))
        memo[m] = best; return best
    v = V(net["target"])
    return v, len(memo)                             # value, number of molecules expanded

def rollout_value(net, stack, rng, depth, maxdepth=12, policy="random"):
    """Finish a partial route: pop the first unsolved molecule, choose a reaction, push its precursors."""
    val = 1.0; stack = list(stack)
    while stack:
        m = stack.pop()
        if net["buyable"][m]: continue
        rs = net["rxns"].get(m, [])
        if not rs or depth >= maxdepth: return 0.0
        y, pre = rs[rng.integers(len(rs))] if policy == "random" else max(rs, key=lambda r: r[0])
        val *= y; stack.extend(pre); depth += 1
    return val

def greedy_route(net):
    return rollout_value(net, [net["target"]], None, 0, policy="greedy")

class Node:
    __slots__ = ("stack", "val", "depth", "children", "N", "W", "untried", "terminal")
    def __init__(self, stack, val, depth, net):
        # strip purchasable molecules from the top of the stack; the next unsolved one decides the actions
        stack = [m for m in stack if not net["buyable"][m]]
        self.stack, self.val, self.depth = stack, val, depth
        self.children, self.N, self.W = {}, 0, 0.0
        self.terminal = (len(stack) == 0) or (not net["rxns"].get(stack[-1], [])) or depth >= 12
        self.untried = [] if self.terminal else list(range(len(net["rxns"][stack[-1]])))

def mcts(net, n_sim, c=0.4, seed=0):
    rng = np.random.default_rng(seed)
    net["rxns"].seen = set()
    root = Node([net["target"]], 1.0, 0, net); best = 0.0
    for _ in range(n_sim):
        node, path = root, [root]
        # 1) selection (UCT)
        while not node.terminal and not node.untried and node.children:
            node = max(node.children.values(),
                       key=lambda ch: ch.W / ch.N + c * math.sqrt(math.log(node.N) / ch.N))
            path.append(node)
        # 2) expansion: apply one untried reaction to the current molecule
        if not node.terminal and node.untried:
            a = node.untried.pop(rng.integers(len(node.untried)))
            y, pre = net["rxns"][node.stack[-1]][a]
            child = Node(node.stack[:-1] + pre, node.val * y, node.depth + 1, net)
            node.children[a] = child; node = child; path.append(node)
        # 3) simulation (random rollout) ; a dead end is worth 0
        if node.terminal:
            v = node.val if len(node.stack) == 0 else 0.0
        else:
            v = node.val * rollout_value(net, node.stack, rng, node.depth)
        best = max(best, v)
        # 4) backup
        for nd in path: nd.N += 1; nd.W += v
    return best, len(net["rxns"].seen)

if __name__ == "__main__":
    t0 = time.time()
    budgets = [10, 30, 100, 300, 1000]
    nets, rows = [], []
    for s in range(400):
        net = make_network(s); v, nexp = dp_best(net)
        if v > 0: nets.append((net, v, nexp))
        if len(nets) == 100: break
    print(f"networks: {len(nets)} solvable (of {s+1} generated); molecules {sum(LEVELS)}, levels {len(LEVELS)}")
    opt = np.array([v for _, v, _ in nets]); nexp = np.array([e for _, _, e in nets])
    print(f"exact DP optimum (product of yields): mean {opt.mean():.3f}, median {np.median(opt):.3f}; "
          f"molecules DP had to expand: mean {nexp.mean():.1f}")
    g = np.array([greedy_route(n) for n, _, _ in nets])
    print(f"greedy (highest-yield step each time): success {np.mean(g > 0):.2f}, value/optimum {np.mean(g / opt):.3f}, "
          f"optimal {np.mean(np.isclose(g, opt)):.2f}")
    def random_search(n, B, seed):
        rng = np.random.default_rng(seed); n["rxns"].seen = set()
        v = max(rollout_value(n, [n["target"]], rng, 0) for _ in range(B)); return v, len(n["rxns"].seen)
    res = {}
    print("budget = number of complete simulations (rollouts); 'queried' = distinct molecules sent to the reaction model")
    for B in budgets:
        for name, fn in [("random rollouts", lambda n, i: random_search(n, B, i)),
                         ("MCTS (UCT c=0.4)", lambda n, i: mcts(n, B, c=0.4, seed=i))]:
            out = [fn(n, i) for i, (n, _, _) in enumerate(nets)]
            vb = np.array([o[0] for o in out]); q = np.array([o[1] for o in out])
            res[(name, B)] = [float(np.mean(vb > 0)), float(np.mean(vb / opt)), float(np.mean(np.isclose(vb, opt))), float(q.mean()),
                              float(np.std(vb / opt) / np.sqrt(len(opt)))]
            print(f"  B={B:5d} {name:17s}: success {res[(name,B)][0]:.2f}, value/optimum {res[(name,B)][1]:.3f} +/- {res[(name,B)][4]:.3f}, "
                  f"optimal {res[(name,B)][2]:.2f}, queried {res[(name,B)][3]:.1f}")
    d = [mcts(n, 300, c=0.4, seed=i)[0] / v - random_search(n, 300, i)[0] / v for i, (n, v, _) in enumerate(nets)]
    print(f"  paired difference MCTS - random at B=300: {np.mean(d):.3f} +/- {np.std(d)/np.sqrt(len(d)):.3f} (SE); "
          f"MCTS better on {np.mean(np.array(d) > 1e-12):.2f}, worse on {np.mean(np.array(d) < -1e-12):.2f} of networks")
    for c in [0.1, 1.4]:
        out = [mcts(n, 300, c=c, seed=i) for i, (n, _, _) in enumerate(nets)]
        vb = np.array([o[0] for o in out])
        print(f"  B=  300 MCTS c={c}: value/optimum {np.mean(vb / opt):.3f}, optimal {np.mean(np.isclose(vb, opt)):.2f}")
    json.dump({f"{k}_{B}": v for (k, B), v in res.items()} | {"greedy": float(np.mean(g / opt))}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "82_mcts.json"), "w"))
    print(f"wall {time.time()-t0:.0f} s")
