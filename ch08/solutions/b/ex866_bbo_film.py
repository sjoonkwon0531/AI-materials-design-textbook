"""Ex 8.6.6: CMA-ES and GP-EI on the film environment as a black box (open-loop 12-step schedule -> one noisy
episode = one 'wafer experiment'); budgets 50, 200, 1,000; 3 repeats; recommended schedule evaluated on the
Lab 8.14 test episodes (2,000, seed 12345)."""
import time, sys, numpy as np, cma
from scipy.stats import norm
from common import lab
E = lab("85_film_env"); P = E.FilmParams(); t0 = time.time()
def wafer(s, seed): return E.rollout_batch(lambda t, o: np.full(len(o), s[t]), 1, seed, P)[0][0]
def test(s): return E.rollout_batch(lambda t, o: np.full(len(o), s[t]), 2000, 12345, P)[0].mean()

def run_cma(budget, rep):
    es = cma.CMAEvolutionStrategy(np.zeros(12), 0.3, dict(bounds=[-1, 1], seed=rep + 1, verbose=-9)); n = 0
    while n < budget:
        X = es.ask(); X = X[:budget - n]
        F = [-wafer(x, 500_000 * (rep + 1) + n + k) for k, x in enumerate(X)]; n += len(X)
        if len(X) == es.popsize: es.tell(X, F)
    return np.asarray(es.result.xfavorite)

def gp_fit(Xa, Yn):
    best = None
    for ls in (0.5, 1.0, 2.0, 4.0):
        K0 = np.exp(-0.5 * ((Xa[:, None] - Xa[None]) ** 2).sum(-1) / ls ** 2)
        for nv in (0.03, 0.1, 0.3, 1.0):
            L = np.linalg.cholesky(K0 + nv * np.eye(len(Xa))); al = np.linalg.solve(L.T, np.linalg.solve(L, Yn))
            ll = -0.5 * Yn @ al - np.log(np.diag(L)).sum()
            if best is None or ll > best[0]: best = (ll, ls, nv, L, al)
    return best[1:]

def gp_pred(Xa, ls, L, al, C):
    Ks = np.exp(-0.5 * ((C[:, None] - Xa[None]) ** 2).sum(-1) / ls ** 2)
    v = np.linalg.solve(L, Ks.T); return Ks @ al, np.sqrt(np.clip(1 - (v ** 2).sum(0), 1e-12, None))

def run_gp(budget, rep):
    rng = np.random.default_rng(700 + rep); q = 1 if budget <= 50 else (5 if budget <= 200 else 20)
    X = list(rng.uniform(-1, 1, (10, 12))); Y = [wafer(x, 600_000 * (rep + 1) + i) for i, x in enumerate(X)]
    while len(Y) < budget:
        Xa, Ya = np.array(X), np.array(Y); mu0, sd0 = Ya.mean(), Ya.std() + 1e-9; Yn = (Ya - mu0) / sd0
        ls, nv, L, al = gp_fit(Xa, Yn)
        m_obs, _ = gp_pred(Xa, ls, L, al, Xa); inc = m_obs.max(); top = Xa[np.argsort(m_obs)[-5:]]
        C = np.vstack([rng.uniform(-1, 1, (1000, 12)), np.clip(top[rng.integers(0, 5, 1000)] + 0.1 * rng.standard_normal((1000, 12)), -1, 1)])
        mu, sd = gp_pred(Xa, ls, L, al, C); z = (mu - inc) / sd; ei = (mu - inc) * norm.cdf(z) + sd * norm.pdf(z)
        for j in np.argsort(ei)[::-1][:min(q, budget - len(Y))]:
            X.append(C[j]); Y.append(wafer(C[j], 600_000 * (rep + 1) + len(Y)))
    Xa, Ya = np.array(X), np.array(Y); Yn = (Ya - Ya.mean()) / Ya.std(); ls, nv, L, al = gp_fit(Xa, Yn)
    m_obs, _ = gp_pred(Xa, ls, L, al, Xa)
    return Xa[int(np.argmax(m_obs))], ls, nv

which = sys.argv[1] if len(sys.argv) > 1 else "both"
for budget in (50, 200, 1000):
    for meth in ("CMA-ES", "GP-EI"):
        if which != "both" and which != meth: continue
        res = []
        for rep in range(1 if (meth == 'GP-EI' and budget == 1000) else 3):
            if meth == "CMA-ES": x = run_cma(budget, rep); extra = ""
            else: x, ls, nv = run_gp(budget, rep); extra = f" (ls {ls:g}, noise {nv:g})"
            res.append(test(x))
            print(f"  {meth} budget {budget:4d} rep {rep}: test J {res[-1]:.3f} | schedule " + " ".join(f"{T:.0f}" for T in E.to_T(x)) + extra + f" | {time.time()-t0:.0f} s", flush=True)
        print(f"{meth:<7s} budget {budget:4d}: test J {np.mean(res):.3f} +/- {np.std(res, ddof=1):.3f} (sd over repeats)", flush=True)
