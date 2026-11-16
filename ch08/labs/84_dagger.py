"""Lab 8.4 (b): compounding errors of behaviour cloning (BC) and the DAgger fix, on the furnace of 84_thermal_env.
Expert: the nominal REINFORCE policy of Lab 8.4 (a) (seed 0, read from 84_domain_rand_results.json).
Learner: a regression tree (min_samples_leaf=5) on the 7 observation features, i.e. a model that cannot
extrapolate (Ch.6 Lab 6.15).  Plant: nominal.  Demonstrations and DAgger rollouts are collected in a quiet
simulator (noise 0.0005); every policy is evaluated in the plant with the standard noise 0.003.  5 seeds.  Usage: python 84_dagger.py   (~1-2 min)"""
import os, json, time, importlib.util
import numpy as np
from sklearn.tree import DecisionTreeRegressor
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("env84", os.path.join(HERE, "84_thermal_env.py"))
E = importlib.util.module_from_spec(_s); _s.loader.exec_module(E)
t0 = time.time()
SCALE = np.array([1, 1, 0.1, 1, 10, 1, 1, 1.0])
th = np.array(json.load(open(os.path.join(HERE, "84_domain_rand_results.json")))["nominal"][0]["theta"])
expert = lambda o: np.hstack([o, np.ones((len(o), 1))]) * SCALE @ th
P = (E.NOM["h"], E.NOM["G"], E.NOM["d"])
DN = 0.0005                                   # demonstration / data-collection noise


def fit(O, U, seed):
    return DecisionTreeRegressor(min_samples_leaf=5, random_state=seed).fit(O, U)


def run(seed, n_ep=2, iters=6, T=E.T_EP, return_models=False):
    """n_ep episodes (T steps each) per round; BC gets all rounds from the expert, DAgger from the learner."""
    rng = np.random.default_rng(seed); out = {"bc": [], "dagger": [], "n": []}
    # --- BC: expert rollouts only
    O_bc, U_bc = [], []
    # --- DAgger: round 0 = expert rollouts, then learner rollouts relabelled by the expert
    O_dg, U_dg = [], []
    for k in range(iters):
        r = E.rollout(expert, *P, n_ep, rng, T=T, record=True, noise=DN)
        o = r["obs"].reshape(-1, 7); O_bc.append(o); U_bc.append(expert(o))
        if k == 0: O_dg.append(o); U_dg.append(expert(o))
        else:
            r = E.rollout(lambda q: pi_dg.predict(q), *P, n_ep, rng, T=T, record=True, noise=DN)
            o = r["obs"].reshape(-1, 7); O_dg.append(o); U_dg.append(expert(o))   # expert labels learner's states
        pi_bc = fit(np.vstack(O_bc), np.hstack(U_bc), seed); pi_dg = fit(np.vstack(O_dg), np.hstack(U_dg), seed)
        ev = np.random.default_rng(10_000 + k)
        out["bc"].append(E.rollout(lambda q: pi_bc.predict(q), *P, 200, ev, T=T)["cost"].mean())
        out["dagger"].append(E.rollout(lambda q: pi_dg.predict(q), *P, 200, np.random.default_rng(10_000 + k), T=T)["cost"].mean())
        out["n"].append(sum(len(u) for u in U_bc))
    # distribution shift diagnostic: distance of learner-visited states to the BC training set (standardised)
    sd = np.vstack(O_bc).std(0) + 1e-9
    r_bc = E.rollout(lambda q: pi_bc.predict(q), *P, 50, np.random.default_rng(7), T=T, record=True)["obs"].reshape(-1, 7)
    r_ex = E.rollout(expert, *P, 50, np.random.default_rng(7), T=T, record=True)["obs"].reshape(-1, 7)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(np.vstack(O_bc) / sd)
    out["d_expert"] = float(np.median(nn.kneighbors(r_ex / sd)[0])); out["d_bc"] = float(np.median(nn.kneighbors(r_bc / sd)[0]))
    out["d_bc_p90"] = float(np.quantile(nn.kneighbors(r_bc / sd)[0], 0.9)); out["d_expert_p90"] = float(np.quantile(nn.kneighbors(r_ex / sd)[0], 0.9))
    if return_models: out["models"] = (pi_bc, pi_dg)
    return out


exp_cost = E.rollout(expert, *P, 1000, np.random.default_rng(3))["cost"].mean()
print("expert on nominal plant: mean |e| %.1f K" % (1e3 * exp_cost))
R = [run(s) for s in range(5)]
n = R[0]["n"]
for key in ("bc", "dagger"):
    A = 1e3 * np.array([r[key] for r in R])
    print("%-7s " % key + " | ".join("n=%5d: %5.1f+/-%4.1f K" % (n[k], A[:, k].mean(), A[:, k].std()) for k in range(len(n))))
print("NN distance to BC data (standardised, median / p90): expert-visited %.3f / %.3f, BC-learner-visited %.3f / %.3f" % (
    np.mean([r["d_expert"] for r in R]), np.mean([r["d_expert_p90"] for r in R]),
    np.mean([r["d_bc"] for r in R]), np.mean([r["d_bc_p90"] for r in R])))
# horizon dependence: same budget of 6 rounds x 2 episodes, longer soak
H = {}
for T in (30, 60, 120, 240):
    rs = [run(s, T=T) for s in range(5)]
    eT = 1e3 * E.rollout(expert, *P, 500, np.random.default_rng(3), T=T)["cost"].mean()
    H[T] = dict(expert=eT, bc=[1e3 * r["bc"][-1] for r in rs], dagger=[1e3 * r["dagger"][-1] for r in rs])
    print("T=%3d: expert %5.1f K | BC %6.1f +/- %5.1f K | DAgger %5.1f +/- %4.1f K  (total cost per episode: BC %.1f, DAgger %.1f, expert %.1f K s)" % (
        T, eT, np.mean(H[T]["bc"]), np.std(H[T]["bc"]), np.mean(H[T]["dagger"]), np.std(H[T]["dagger"]),
        T * np.mean(H[T]["bc"]), T * np.mean(H[T]["dagger"]), T * eT), flush=True)
json.dump(dict(runs=R, horizon=H, expert=exp_cost), open(os.path.join(HERE, "84_dagger_results.json"), "w"))
print("wall %.0f s" % (time.time() - t0))
