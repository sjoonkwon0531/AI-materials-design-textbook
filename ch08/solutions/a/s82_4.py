"""Ex 8.2.4: softmax (temperature tau) rollout policy for MCTS / random search; deeper networks."""
import sys, importlib, math, time, numpy as np
sys.path.insert(0, "../ch08/labs")
M = importlib.import_module("82_mcts")
MAXD = 12

def pick(rs, rng, tau):
    if tau == np.inf: return rs[rng.integers(len(rs))]
    y = np.array([r[0] for r in rs]); w = np.exp((np.log(y) - np.log(y).max()) / tau); w /= w.sum()
    return rs[rng.choice(len(rs), p=w)]

def rollout_value(net, stack, rng, depth, tau):
    val = 1.0; stack = list(stack)
    while stack:
        m = stack.pop()
        if net["buyable"][m]: continue
        rs = net["rxns"].get(m, [])
        if not rs or depth >= MAXD: return 0.0
        y, pre = pick(rs, rng, tau); val *= y; stack.extend(pre); depth += 1
    return val

class Node:
    __slots__ = ("stack", "val", "depth", "children", "N", "W", "untried", "terminal")
    def __init__(self, stack, val, depth, net):
        stack = [m for m in stack if not net["buyable"][m]]
        self.stack, self.val, self.depth = stack, val, depth
        self.children, self.N, self.W = {}, 0, 0.0
        self.terminal = (len(stack) == 0) or (not net["rxns"].get(stack[-1], [])) or depth >= MAXD
        self.untried = [] if self.terminal else list(range(len(net["rxns"][stack[-1]])))

def mcts(net, n_sim, tau, c=0.4, seed=0):
    rng = np.random.default_rng(seed); root = Node([net["target"]], 1.0, 0, net); best = 0.0
    for _ in range(n_sim):
        node, path = root, [root]
        while not node.terminal and not node.untried and node.children:
            node = max(node.children.values(), key=lambda ch: ch.W / ch.N + c * math.sqrt(math.log(node.N) / ch.N)); path.append(node)
        if not node.terminal and node.untried:
            a = node.untried.pop(rng.integers(len(node.untried)))
            y, pre = net["rxns"][node.stack[-1]][a]
            ch = Node(node.stack[:-1] + pre, node.val * y, node.depth + 1, net); node.children[a] = ch; node = ch; path.append(node)
        v = (node.val if not node.stack else 0.0) if node.terminal else node.val * rollout_value(net, node.stack, rng, node.depth, tau)
        best = max(best, v)
        for nd in path: nd.N += 1; nd.W += v
    return best

def random_search(net, B, tau, seed):
    rng = np.random.default_rng(seed); return max(rollout_value(net, [net["target"]], rng, 0, tau) for _ in range(B))

def route_size(net):
    """number of reactions in the DP-optimal route (multiplicity counted, as the rollout counts depth)"""
    memo = {}
    def V(m):
        if m in memo: return memo[m]
        if net["buyable"][m]: memo[m] = (1.0, 0); return memo[m]
        best = (0.0, 0)
        for y, pre in net["rxns"].get(m, []):
            sub = [V(p) for p in pre]; v = y * math.prod(s[0] for s in sub)
            if v > best[0]: best = (v, 1 + sum(s[1] for s in sub))
        memo[m] = best; return best
    return V(net["target"])[1]

def run(levels, maxd, label, budgets=(30, 100), taus=(0.1, 0.3, 1.0, np.inf)):
    global MAXD; MAXD = maxd; M.LEVELS = levels
    nets = []
    for s in range(2000):
        net = M.make_network(s); v, _ = M.dp_best(net)
        if v > 0: nets.append((net, v))
        if len(nets) == 100: break
    opt = np.array([v for _, v in nets]); sizes = np.array([route_size(n) for n, _ in nets])
    g = np.array([M.greedy_route(n) for n, _ in nets])
    print(f"[{label}] levels {levels}, solvable 100 of {s+1}; DP optimum mean {opt.mean():.4f} median {np.median(opt):.4f}; "
          f"optimal-route reactions mean {sizes.mean():.1f}, max {sizes.max()}, >{maxd}: {(sizes>maxd).mean():.2f}; depth cap {maxd}; greedy value/opt {np.mean(g/opt):.3f}")
    for B in budgets:
        for tau in taus:
            r = np.array([random_search(n, B, tau, i) / v for i, (n, v) in enumerate(nets)])
            mc = np.array([mcts(n, B, tau, seed=i) / v for i, (n, v) in enumerate(nets)])
            d = mc - r
            print(f"  B={B:4d} tau={tau:>4}: random {r.mean():.3f}+/-{r.std()/10:.3f} (opt {np.isclose(r,1).mean():.2f}) | "
                  f"MCTS {mc.mean():.3f}+/-{mc.std()/10:.3f} (opt {np.isclose(mc,1).mean():.2f}) | diff {d.mean():+.3f}+/-{d.std()/10:.3f}")
        sys.stdout.flush()

if __name__ == "__main__":
    t0 = time.time()
    run([30, 25, 25, 20, 15, 10, 1], 12, "original (levels 0-6)")
    print(f"[{time.time()-t0:.0f} s]")
    run([30, 25, 25, 20, 20, 15, 15, 10, 10, 1], 12, "deep (levels 0-9), cap 12")
    print(f"[{time.time()-t0:.0f} s]")
    run([30, 25, 25, 20, 20, 15, 15, 10, 10, 1], 40, "deep (levels 0-9), cap 40")
    print(f"wall {time.time()-t0:.0f} s")
