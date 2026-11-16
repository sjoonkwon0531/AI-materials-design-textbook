"""82_compenv.py -- surrogate band-gap model + composition-editing Gymnasium environment
(Ch.8 Sec.8.2.1, Lab 8.2.1; owner D may import it for Sec.8.5).

Contents
  featurize(formulas)            -> Magpie 132 features (matminer, offline), identical column order
                                    to ch8/data/magpie_expt_gap.parquet
  Surrogate                      -> deep ensemble of M small MLPs (L1 loss) on standardized Magpie
                                    features; .predict(X) returns (mean, std, members)
  train_surrogate(...)           -> fit and cache to labs/82_surrogate.pt
  GridTable                      -> all compositions with N atoms over k elements, with surrogate
                                    mean/std, 5-NN distance to training set, charge balance
  CompositionEnv(gym.Env)        -> state = integer counts (MultiDiscrete), action = move one atom
                                    from element i to element j (k(k-1) actions) or stay, T steps/episode,
                                    reward = potential-based difference of a score:
                                       naive     : s(x) = -|mu(x) - target|
                                       penalized : s(x) = -|mu(x) - target| - kappa*sigma(x)
                                                          - lam*max(0, d(x)/d95 - 1) - cb*[not balanced]
Usage:  import importlib; m = importlib.import_module("82_compenv")  (file name starts with a digit)
"""
import os, itertools, warnings
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# formal oxidation states used for the charge-balance plausibility check
OXI = {"Cu": 1, "In": 3, "Ga": 3, "S": -2, "Se": -2, "Te": -2, "Zn": 2, "Cd": 2, "O": -2}

_EP = None
def featurize(formulas):
    """Magpie ElementProperty features (132) for a list of formula strings or Composition objects."""
    global _EP
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    if _EP is None:
        _EP = ElementProperty.from_preset("magpie", impute_nan=False)
    return np.array([_EP.featurize(Composition(f) if isinstance(f, str) else f) for f in formulas],
                    dtype=np.float64)

def load_data():
    import pandas as pd
    d = pd.read_parquet(os.path.join(DATA, "magpie_expt_gap.parquet"))
    return d["formula"].tolist(), d.iloc[:, 2:].to_numpy(np.float64), d["target"].to_numpy(np.float64)

def make_mlp(p, h=128):
    return nn.Sequential(nn.Linear(p, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

class Surrogate:
    """Deep ensemble (Lakshminarayanan et al. 2017 style: random init + shuffling only)."""
    def __init__(self, mu, sd, nets):
        self.mu, self.sd, self.nets = mu, sd, nets
    def scale(self, X):
        return np.clip((X - self.mu) / self.sd, -6, 6).astype(np.float32)
    @torch.no_grad()
    def predict(self, X):
        Z = torch.from_numpy(self.scale(np.asarray(X, np.float64)))
        P = np.stack([net(Z).squeeze(-1).numpy() for net in self.nets])     # (M, n)
        P = np.clip(P, 0.0, None)                                         # band gap >= 0
        return P.mean(0), P.std(0), P

def fit_ensemble(X, y, M=5, epochs=60, bs=128, lr=2e-3, wd=1e-4, seed=0):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    sur = Surrogate(mu, sd, [])
    Z = torch.from_numpy(sur.scale(X)); Y = torch.from_numpy(y.astype(np.float32))
    n = len(y); nb = int(np.ceil(n / bs))
    for m in range(M):
        torch.manual_seed(1000 * seed + m)
        net = make_mlp(X.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb, pct_start=0.1)
        g = torch.Generator().manual_seed(1000 * seed + m)
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g)
            for b in range(nb):
                idx = perm[b * bs:(b + 1) * bs]
                loss = (net(Z[idx]).squeeze(-1) - Y[idx]).abs().mean()     # L1, as in Lab 7.18
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval(); sur.nets.append(net)
    return sur

def train_surrogate(path=os.path.join(HERE, "82_surrogate.pt"), M=5, seed=0, force=False):
    _, X, y = load_data()
    if os.path.exists(path) and not force:
        st = torch.load(path, weights_only=False)
        nets = []
        for sdict in st["nets"]:
            net = make_mlp(X.shape[1]); net.load_state_dict(sdict); net.eval(); nets.append(net)
        return Surrogate(st["mu"], st["sd"], nets)
    sur = fit_ensemble(X, y, M=M, seed=seed)
    torch.save({"mu": sur.mu, "sd": sur.sd, "nets": [n.state_dict() for n in sur.nets]}, path)
    return sur

class KNNDistance:
    """Mean distance to the k nearest training points in the surrogate's standardized space
    (the applicability-domain test of Lab 6.15). d95 = 95th percentile of leave-one-out training distances."""
    def __init__(self, sur, Xtr, k=5):
        from sklearn.neighbors import NearestNeighbors
        self.sur, self.k = sur, k
        Ztr = sur.scale(Xtr)
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(Ztr)
        dtr, _ = self.nn.kneighbors(Ztr)
        self.d95 = float(np.percentile(dtr[:, 1:].mean(1), 95))
    def __call__(self, X):
        d, _ = self.nn.kneighbors(self.sur.scale(X), n_neighbors=self.k)
        return d.mean(1)

def grid_states(k, N):
    """All integer count vectors of length k with sum N and at least two nonzero entries."""
    out = []
    for c in itertools.combinations(range(N + k - 1), k - 1):    # stars and bars
        b = (-1,) + c + (N + k - 1,)
        v = tuple(b[i + 1] - b[i] - 1 for i in range(k))
        if sum(x > 0 for x in v) >= 2:
            out.append(v)
    return out

def formula(elements, v):
    return "".join(f"{e}{n}" for e, n in zip(elements, v) if n > 0)

def charge_balanced(elements, v):
    return sum(OXI[e] * n for e, n in zip(elements, v)) == 0

class GridTable:
    """Precomputed surrogate outputs on the whole composition grid (cheap: a few thousand formulas).
    In a real campaign each entry would be a surrogate call; precomputing only saves time."""
    def __init__(self, elements=("Cu", "In", "Ga", "S", "Se"), N=12, sur=None, cache=True):
        self.elements, self.N = tuple(elements), N
        self.states = grid_states(len(elements), N)
        self.index = {s: i for i, s in enumerate(self.states)}
        tag = "".join(elements) + f"_N{N}"
        fpath = os.path.join(HERE, f"82_grid_{tag}.npz")
        sur = sur or train_surrogate()
        if cache and os.path.exists(fpath):
            z = np.load(fpath); self.mu, self.sigma, self.dist, self.d95 = z["mu"], z["sigma"], z["dist"], float(z["d95"])
            self.members = z["members"]
        else:
            X = featurize([formula(self.elements, s) for s in self.states])
            self.mu, self.sigma, self.members = sur.predict(X)
            _, Xtr, _ = load_data()
            knn = KNNDistance(sur, Xtr); self.dist = knn(X); self.d95 = knn.d95
            np.savez(fpath, mu=self.mu, sigma=self.sigma, dist=self.dist, d95=self.d95, members=self.members)
        self.balanced = np.array([charge_balanced(self.elements, s) for s in self.states])

    def score(self, target, kappa=0.0, lam=0.0, cb=0.0):
        """s(x) = -|mu-target| - kappa*sigma - lam*max(0, d/d95-1) - cb*[not charge balanced]."""
        s = -np.abs(self.mu - target) - kappa * self.sigma - lam * np.maximum(0.0, self.dist / self.d95 - 1.0)
        return s - cb * (~self.balanced)

class CompositionEnv(gym.Env):
    """Sequential composition editing on a simplex grid.
    observation: counts (MultiDiscrete [N+1]*k);  action a -> (i, j): move one atom from element i to j;
    the last action (index k(k-1)) is "stay".
    A move from an empty element, or one that leaves fewer than two elements, is a no-op.
    reward r_t = s(x_{t+1}) - s(x_t) (potential-based), so the return equals s(x_T) - s(x_0)."""
    metadata = {"render_modes": []}

    def __init__(self, table, target=1.4, T=8, kappa=0.0, lam=0.0, cb=0.0, start=None, seed=None):
        super().__init__()
        self.tab, self.target, self.T = table, target, T
        self.k, self.N = len(table.elements), table.N
        self.pairs = [(i, j) for i in range(self.k) for j in range(self.k) if i != j] + [(None, None)]  # last = stay
        self.action_space = spaces.Discrete(len(self.pairs))
        self.observation_space = spaces.MultiDiscrete([self.N + 1] * self.k)
        self.S = table.score(target, kappa, lam, cb)
        self.start = start                                    # None -> uniformly random grid state
        self._rng = np.random.default_rng(seed)

    def _obs(self):
        return np.array(self.state, dtype=np.int64)

    def _info(self):
        i = self.tab.index[self.state]
        return {"idx": i, "mu": float(self.tab.mu[i]), "sigma": float(self.tab.sigma[i]),
                "dist": float(self.tab.dist[i]), "balanced": bool(self.tab.balanced[i])}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.start is None:
            self.state = self.tab.states[self._rng.integers(len(self.tab.states))]
        else:
            self.state = tuple(self.start)
        self.t = 0
        return self._obs(), self._info()

    def next_state(self, state, a):
        i, j = self.pairs[a]
        if i is None:
            return state
        v = list(state)
        if v[i] == 0:
            return state
        v[i] -= 1; v[j] += 1
        v = tuple(v)
        return v if v in self.tab.index else state

    def step(self, a):
        s0 = self.S[self.tab.index[self.state]]
        self.state = self.next_state(self.state, int(a))
        self.t += 1
        r = float(self.S[self.tab.index[self.state]] - s0)
        return self._obs(), r, False, self.t >= self.T, self._info()
