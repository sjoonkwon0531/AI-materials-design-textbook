"""Ex 8.5.3: PPO with the first set-point frozen at 650 K (action -0.5); 3 seeds x 240,000 steps."""
import os, sys, time, types
import numpy as np, torch, torch.nn as nn, gymnasium as gym
from common import lab, LABS
torch.set_num_threads(1)
E = lab("85_film_env"); P = E.FilmParams()
src = open(os.path.join(LABS, "85_ppo.py")).read()
sec3 = src.split("# ---------------------------------------------------------------- 3. PPO from scratch")[1].split(
    "# ---------------------------------------------------------------- 4. evaluation")[0]

class FixFirst(gym.ActionWrapper):
    def __init__(self, env, a0=-0.5):
        super().__init__(env); self.a0 = a0
    def action(self, act):
        return np.array([self.a0], np.float32) if self.env.unwrapped.t == 0 else act

E2 = types.SimpleNamespace(FilmGrowthEnv=lambda: FixFirst(E.FilmGrowthEnv()), OpenLoop=E.OpenLoop)
ns = dict(np=np, torch=torch, nn=nn, gym=gym, E=E2, P=P)
exec(sec3, ns)
t0 = time.time(); Js = []
lam_tb = lambda T: P.lam * P.tb(T)
rng = np.random.default_rng(0); d = rng.uniform(-40, 40, 200000); eps = 3 * rng.standard_normal(200000)
print(f"direct budget cost of the first step: lam*E[tb]: 600 K {lam_tb(600+d+eps).mean():.4f}, 650 K {lam_tb(650+d+eps).mean():.4f}")
for s in range(3):
    ac, curve = ns["ppo"](s)
    f = ns["policy_fn"](ac)
    J, w, TB, T, dd = E.rollout_batch(lambda t, o: np.full(len(o), -0.5) if t == 0 else f(t, o), 2000, 12345, P, return_traj=True)
    Tl = (T - dd[:, None])[:, 4:].mean(1)
    Js.append(J.mean())
    print(f"seed {s}: J = {J.mean():.3f} +/- {J.std(ddof=1)/np.sqrt(2000):.3f} | w_N {w.mean():.3f} | TB {TB.mean():.2f} | 5th pct {np.percentile(J,5):.3f} | "
          f"corr(set-point 5-12, delta) {np.corrcoef(Tl, dd)[0,1]:+.2f} slope {np.polyfit(dd, Tl, 1)[0]:+.2f} | "
          f"mean set-point by step " + " ".join(f"{x:.0f}" for x in (T - dd[:, None]).mean(0)) + f" | {time.time()-t0:.0f} s", flush=True)
    m = np.abs(dd) > 30
    print(f"     |delta|>30 K: J {J[m].mean():.3f}; |delta|<5 K: J {J[np.abs(dd)<5].mean():.3f}", flush=True)
print(f"PPO (first step fixed at 650 K) over 3 seeds: J = {np.mean(Js):.3f} +/- {np.std(Js, ddof=1):.3f}")
# reference: the stored Lab PPO models, by |delta|
R = np.load(os.path.join(LABS, "85_ppo_results.npz"))
for s in range(3):
    J = R[f"PPO seed {s}__J"]; dd = R[f"PPO seed {s}__d"]
    print(f"Lab PPO seed {s}: J {J.mean():.3f}; |delta|>30 K: {J[np.abs(dd)>30].mean():.3f}; |delta|<5 K: {J[np.abs(dd)<5].mean():.3f}")
