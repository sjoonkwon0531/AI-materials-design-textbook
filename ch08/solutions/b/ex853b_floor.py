"""Ex 8.5.3 supplement: forbid cold probing anywhere -- every set-point clipped to >= 650 K (action >= -0.5); 2 seeds."""
import os, time, types, numpy as np, torch, torch.nn as nn, gymnasium as gym
from common import lab, LABS
torch.set_num_threads(1)
E = lab("85_film_env"); P = E.FilmParams()
src = open(os.path.join(LABS, "85_ppo.py")).read()
sec3 = src.split("# ---------------------------------------------------------------- 3. PPO from scratch")[1].split(
    "# ---------------------------------------------------------------- 4. evaluation")[0]
class Floor(gym.ActionWrapper):
    def action(self, act): return np.clip(act, -0.5, 1.0).astype(np.float32)
E2 = types.SimpleNamespace(FilmGrowthEnv=lambda: Floor(E.FilmGrowthEnv()), OpenLoop=E.OpenLoop)
ns = dict(np=np, torch=torch, nn=nn, gym=gym, E=E2, P=P); exec(sec3, ns); t0 = time.time()
for s in range(2):
    ac, _ = ns["ppo"](s); f = ns["policy_fn"](ac)
    J, w, TB, T, dd = E.rollout_batch(lambda t, o: np.clip(f(t, o), -0.5, 1), 2000, 12345, P, return_traj=True)
    Tl = (T - dd[:, None])[:, 4:].mean(1)
    print(f"floor 650 K seed {s}: J = {J.mean():.3f} +/- {J.std(ddof=1)/np.sqrt(2000):.3f} | w_N {w.mean():.3f} | TB {TB.mean():.2f} | 5th pct {np.percentile(J,5):.3f} | "
          f"corr {np.corrcoef(Tl, dd)[0,1]:+.2f} slope {np.polyfit(dd, Tl, 1)[0]:+.2f} | set-point by step " + " ".join(f"{x:.0f}" for x in (T - dd[:, None]).mean(0))
          + f" | |d|>30: {J[np.abs(dd)>30].mean():.3f} | {time.time()-t0:.0f} s", flush=True)
