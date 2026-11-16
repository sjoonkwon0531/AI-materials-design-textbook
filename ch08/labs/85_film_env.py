"""85_film_env.py -- Ch.8 Sec.8.5 (owner D): a Gymnasium thin-film growth environment.

Physics (coarse-grained limit of a solid-on-solid KMC, cf. Ch.5 Sec.5.3-5.4):
  linear stochastic growth equation for the height profile h(x,t) on a ring of L sites,
      dh/dt = F - kappa(T) d^4 h/dx^4 - a d^2 h/dx^2 + eta,   <eta eta'> = F delta(x-x') delta(t-t')
  * kappa(T) = kappa_ref * exp[-(E_s/k_B)(1/T - 1/T_ref)]   surface diffusion (Mullins) smoothing
  * a > 0 : step-edge (Ehrlich-Schwoebel) barrier -> uphill current -> long-wavelength (mound) instability
  * eta   : shot noise of random deposition
  In an orthonormal real Fourier basis every mode c_m is an independent Ornstein-Uhlenbeck process with
      Gamma_m(T) = kappa(T) qt_m^4 - a qt_m^2,   qt_m^2 = 4 sin^2(pi m / L)   (lattice Laplacian eigenvalue)
  and over one deposition interval dt with constant T it is integrated EXACTLY:
      c <- c exp(-Gamma dt) + sqrt(F (1 - exp(-2 Gamma dt)) / (2 Gamma)) * xi .
  roughness w = sqrt( (1/L) sum_m c_m^2 ).

Thermal budget (dopant / interface interdiffusion), a steeper Arrhenius law:
      tb_t = dt * exp[-(E_tb/k_B)(1/T - 1/T_ref)]

Episode: N steps, each deposits 1 monolayer (F dt = 1). Action = set-point in [-1, 1] -> T_set = 700 + 100 a [K].
Actual substrate temperature T = T_set + delta + jitter, delta ~ U(-D, D) per episode (unknown thermal contact),
jitter ~ N(0, 3 K) per step.  The agent never sees delta or T; it sees the film.
Observation (float32, 5): [t/N, log(w_all + 1e-3), log(w_low + 1e-3), log(w_high + 1e-3), last action]
   w_low  : roughness of modes m <= 8 (wavelength >= L/8),  w_high : modes m > 8.
Reward r_t = -(w_{t+1} - w_t) - lam * tb_t   ->  return = -w_N - lam * sum_t tb_t   (w_0 = 0).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

KB = 8.617333e-5  # eV/K


class FilmParams:
    def __init__(self, L=128, N=12, kappa_ref=1.0, Es=1.0, a=0.25, Etb=3.0, lam=0.02, Tref=700.0,
                 delta=40.0, jitter=3.0, m_split=8):
        self.L, self.N, self.kappa_ref, self.Es, self.a = L, N, kappa_ref, Es, a
        self.Etb, self.lam, self.Tref, self.delta, self.jitter, self.m_split = Etb, lam, Tref, delta, jitter, m_split
        m = np.concatenate([np.arange(1, L // 2), np.arange(1, L // 2), [L // 2]])  # cos m, sin m, Nyquist
        self.m = m
        self.q2 = 4.0 * np.sin(np.pi * m / L) ** 2
        self.low = m <= m_split

    def kappa(self, T):
        return self.kappa_ref * np.exp(-(self.Es / KB) * (1.0 / T - 1.0 / self.Tref))

    def tb(self, T):
        return np.exp(-(self.Etb / KB) * (1.0 / T - 1.0 / self.Tref))


def ou_step(c, T, P, rng, dt=1.0, F=1.0):
    """Exact OU update of all modes over dt at temperature T (T may be an array over a batch: c is (B, M))."""
    T = np.asarray(T, dtype=float)
    G = P.kappa(T)[..., None] * P.q2 ** 2 - P.a * P.q2           # (B, M) or (M,)
    e = np.exp(-G * dt)
    small = np.abs(G * dt) < 1e-8
    var = np.where(small, F * dt, F * (1.0 - e ** 2) / (2.0 * np.where(small, 1.0, G)))
    return c * e + np.sqrt(var) * rng.standard_normal(c.shape)


def rough(c, P, mask=None):
    cc = c if mask is None else c[..., mask]
    return np.sqrt((cc ** 2).sum(-1) / P.L)


class OpenLoop(gym.ObservationWrapper):
    """Ablation: the agent sees only the clock t/N (film features and last action zeroed)."""
    def observation(self, obs):
        o = np.zeros_like(obs); o[0] = obs[0]; return o


def to_T(action):
    return 700.0 + 100.0 * np.clip(action, -1.0, 1.0)


class FilmGrowthEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params=None, delta=None):
        super().__init__()
        self.P = params or FilmParams()
        self.fixed_delta = delta                          # None -> random per episode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        lo = np.array([0.0, -7.0, -7.0, -7.0, -1.0], dtype=np.float32)    # log(1e-3) = -6.9
        hi = np.array([1.0, 10.0, 10.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    def _obs(self):
        P = self.P
        w, wl, wh = rough(self.c, P), rough(self.c, P, P.low), rough(self.c, P, ~P.low)
        return np.array([self.t / P.N, np.log(w + 1e-3), np.log(wl + 1e-3), np.log(wh + 1e-3), self.last_a],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        P = self.P
        self.delta = (self.np_random.uniform(-P.delta, P.delta) if self.fixed_delta is None
                      else float(self.fixed_delta))
        self.c = np.zeros(len(P.m))
        self.t, self.last_a, self.w, self.TB = 0, 0.0, 0.0, 0.0
        return self._obs(), {"delta": self.delta}

    def step(self, action):
        P = self.P
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        T = to_T(a) + self.delta + P.jitter * self.np_random.standard_normal()
        self.c = ou_step(self.c, T, P, self.np_random)
        tb = float(P.tb(T))
        w_new = float(rough(self.c, P))
        r = -(w_new - self.w) - P.lam * tb
        self.w, self.TB, self.last_a = w_new, self.TB + tb, a
        self.t += 1
        terminated = self.t >= P.N
        return self._obs(), float(r), terminated, False, {"T": T, "tb": tb, "w": w_new, "TB": self.TB}


def rollout_batch(schedule_fn, n, seed, P=None, delta=None, return_traj=False):
    """Vectorized evaluation of a (possibly closed-loop) controller on n episodes with common random numbers.
    schedule_fn(t, obs) -> actions (n,) in [-1,1]; obs is (n, 5) as in FilmGrowthEnv.
    Returns J (n,), final w (n,), TB (n,) [, temperatures (n, N)]."""
    P = P or FilmParams()
    rng = np.random.default_rng(seed)
    d = rng.uniform(-P.delta, P.delta, n) if delta is None else np.full(n, float(delta))
    c = np.zeros((n, len(P.m))); TB = np.zeros(n); last = np.zeros(n); Ts = []
    for t in range(P.N):
        w, wl, wh = rough(c, P), rough(c, P, P.low), rough(c, P, ~P.low)
        obs = np.stack([np.full(n, t / P.N), np.log(w + 1e-3), np.log(wl + 1e-3), np.log(wh + 1e-3), last], 1)
        a = np.clip(np.asarray(schedule_fn(t, obs), dtype=float).reshape(n), -1, 1)
        T = to_T(a) + d + P.jitter * rng.standard_normal(n)
        c = ou_step(c, T, P, rng)
        TB += P.tb(T); last = a; Ts.append(T)
    wN = rough(c, P)
    J = -wN - P.lam * TB
    if return_traj:
        return J, wN, TB, np.stack(Ts, 1), d
    return J, wN, TB
