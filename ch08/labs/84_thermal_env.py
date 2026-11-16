"""Ch.8 Sec.8.4: a vectorised toy rapid-thermal-annealing (RTA) furnace, first-order plus dead time.

    x_{t+1} = x_t + dt/C * ( G * u_{t-d} - h * x_t ) + w_t ,   u in [0, 1]
x = (T - T_amb) / 1000 K, so x = 1 is 1000 K above ambient.  Nominal: C = 1, h = 0.1 /s (tau = C/h = 10 s),
G = 0.2 (u = 0.5 holds x = 1), dead time d = 2 steps, dt = 1 s, process noise w ~ N(0, 0.003^2).
Target profile x*(t): linear ramp 0 -> 1 over 20 s, then a 40 s soak at 1 (episode: 60 steps).
Observation (7): [e_t, e_{t-1}, I_t (clipped integral of e), x*_{t+1}, x*_{t+1} - x*_t, u_{t-1}, u_{t-2}]
with e = x* - x (measured with sensor noise 0.003).  Cost per step: |e_t| + 5 * max(0, x_t - 1.05) (overshoot).
All quantities are synthetic; the model only mimics the structure (lag + dead time + saturation) of RTA."""
import numpy as np

T_EP, RAMP = 60, 20
NOM = dict(h=0.1, G=0.2, d=2)


def target(t):
    t = np.asarray(t, float)
    return np.where(t < RAMP, t / RAMP, 1.0)


class FurnaceBatch:
    """n parallel episodes; h, G, d may be arrays of length n (one plant per episode)."""
    def __init__(self, n, h, G, d, rng, noise=0.003, T=T_EP, x0=None):
        self.n, self.T, self.rng, self.noise = n, T, rng, noise
        self.h = np.broadcast_to(np.asarray(h, float), (n,)).copy()
        self.G = np.broadcast_to(np.asarray(G, float), (n,)).copy()
        self.d = np.broadcast_to(np.asarray(d, int), (n,)).copy()
        self.x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
        self.ubuf = np.zeros((n, 8))                     # ubuf[:, k] = u_{t-1-k}
        self.t, self.I = 0, np.zeros(n)
        self.e_prev = target(0) - self.x

    def obs(self):
        xm = self.x + self.noise * self.rng.standard_normal(self.n)
        e = target(self.t) - xm
        return np.stack([e, self.e_prev, self.I, np.full(self.n, target(self.t + 1)),
                         np.full(self.n, target(self.t + 1) - target(self.t)), self.ubuf[:, 0], self.ubuf[:, 1]], 1), e

    def step(self, u):
        u = np.clip(u, 0.0, 1.0)
        self.ubuf = np.roll(self.ubuf, 1, axis=1); self.ubuf[:, 0] = u
        u_eff = self.ubuf[np.arange(self.n), self.d - 1]          # u_{t-d} (d >= 1)
        self.x = self.x + (self.G * u_eff - self.h * self.x) + self.noise * self.rng.standard_normal(self.n)
        self.t += 1
        e_true = target(self.t) - self.x
        cost = np.abs(e_true) + 5.0 * np.maximum(0.0, self.x - 1.05)
        return cost


def rollout(policy, h, G, d, n, rng, T=T_EP, record=False, noise=0.003):
    """policy(obs) -> u (n,).  Returns per-episode mean cost, max overshoot, and optionally trajectories."""
    env = FurnaceBatch(n, h, G, d, rng, T=T, noise=noise)
    C, X, O, xmax = np.zeros(n), [], [], np.zeros(n)
    for t in range(T):
        o, e = env.obs(); u = policy(o)
        env.I = np.clip(env.I + e, -10, 10); env.e_prev = e
        C += env.step(u); xmax = np.maximum(xmax, env.x)
        if record: X.append(env.x.copy()); O.append(o)
    out = dict(cost=C / T, xmax=xmax)
    if record: out["x"] = np.array(X).T; out["obs"] = np.array(O)
    return out
