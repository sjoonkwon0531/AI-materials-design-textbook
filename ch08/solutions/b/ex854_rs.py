"""Ex 8.5.4: random search budget split (25,000 episodes): n_eval = 1, 10, 100, and a two-stage variant."""
import time, numpy as np
from common import lab
E = lab("85_film_env"); P = E.FilmParams(); t0 = time.time()
def score(s, n, seed): return E.rollout_batch(lambda t, o: np.full(len(o), s[t]), n, seed, P)[0].mean()
def test(s): return E.rollout_batch(lambda t, o: np.full(len(o), s[t]), 2000, 12345, P)[0].mean()
# per-episode spread for a typical schedule (data-like constant 700 K and a random candidate)
c = np.random.default_rng(3).uniform(-1, 1, (1, 12))[0]
for nm, s in (("constant 700 K", np.zeros(12)), ("random candidate #0", c)):
    J = E.rollout_batch(lambda t, o: np.full(len(o), s[t]), 2000, 4, P)[0]
    print(f"per-episode sd of J, {nm}: {J.std():.3f} (mean {J.mean():.3f})")
for pool in (3, 4):
    cands_all = np.random.default_rng(pool).uniform(-1, 1, (25000, P.N))
    for n_eval in (1, 10, 100):
        nc = 25000 // n_eval; cands = cands_all[:nc]
        sc = np.array([score(s, n_eval, 100 + i) for i, s in enumerate(cands)])
        b = int(np.argmax(sc)); tj = test(cands[b])
        q = np.sort(sc)
        print(f"pool {pool} n_eval {n_eval:3d} ({nc:5d} cand): best train {sc[b]:7.3f} | test {tj:7.3f} | gap {sc[b]-tj:6.3f} | "
              f"schedule max {E.to_T(cands[b]).max():.0f} K | {time.time()-t0:.0f} s", flush=True)
    # two-stage: 2,000 x 10 then top 50 x 100
    sc = np.array([score(s, 10, 100 + i) for i, s in enumerate(cands_all[:2000])]); top = np.argsort(sc)[-50:]
    sc2 = np.array([score(cands_all[i], 100, 50000 + i) for i in top]); b = top[int(np.argmax(sc2))]; tj = test(cands_all[b])
    print(f"pool {pool} two-stage (2000x10 + 50x100): best stage-2 score {sc2.max():7.3f} | test {tj:7.3f} | gap {sc2.max()-tj:6.3f}", flush=True)
