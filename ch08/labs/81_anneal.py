"""Lab 81_anneal: annealing-schedule MDP solved exactly (value iteration, policy iteration)
and learned from samples (MC vs TD(0) policy evaluation, Q-learning), checked against DP.
Runtime ~1-2 min on one core. Deterministic seeds."""
import numpy as np, time, importlib
E = importlib.import_module("81_anneal_env")
P, R = E.build_model(); G = E.GAMMA; nS, nA = E.nS, E.nA

def q_from_v(V):
    return R + G * np.einsum("asp,p->sa", P, V)

def policy_eval_exact(pi):
    """Solve (I - gamma P_pi) V = R_pi  (a linear system, Ch.1 §1.4)."""
    Ppi = P[pi, np.arange(nS)]; Rpi = R[np.arange(nS), pi]
    return np.linalg.solve(np.eye(nS) - G * Ppi, Rpi)

# ---------- (a) value iteration = fixed-point iteration of the Bellman optimality operator
def value_iteration(tol=1e-12, kmax=5000):
    V = np.zeros(nS); hist = []
    for k in range(kmax):
        Vn = q_from_v(V).max(1)
        hist.append(np.abs(Vn - V).max()); V = Vn
        if hist[-1] < tol: break
    return V, np.array(hist)

# ---------- (b) policy iteration
def policy_iteration():
    pi = np.zeros(nS, int); it = 0
    while True:
        V = policy_eval_exact(pi); it += 1
        pin = q_from_v(V).argmax(1)
        if np.array_equal(pin, pi): return pi, V, it
        pi = pin

# ---------- (c) model-free evaluation of a fixed policy: MC vs TD(0)
def sample_episode(pi, n0, H, rng):
    s = n0; traj = []
    for t in range(H):
        a = pi[s]; s2, r = E.step(s, a, rng); traj.append((s, r, s2)); s = s2
    return traj

def mc_eval(pi, n_ep, H, rng, t_use=50):
    """Every-visit MC. Episodes of H steps; only visits with t < t_use are used, so each
    return covers >= H - t_use steps and the truncation bias is <= gamma^(H-t_use) |V|max."""
    S = np.zeros(nS); C = np.zeros(nS)
    for _ in range(n_ep):
        traj = sample_episode(pi, rng.integers(0, nS), H, rng); g = 0.0
        for t in range(H - 1, -1, -1):
            s, r, s2 = traj[t]; g = r + G * g
            if t < t_use: S[s] += g; C[s] += 1
    return np.where(C > 0, S / np.maximum(C, 1), 0.0)

def td0_eval(pi, n_ep, H, rng, alpha=0.1):
    """TD(0) with constant step size, V initialised to 0, on the same kind of episodes."""
    V = np.zeros(nS)
    for _ in range(n_ep):
        for (s, r, s2) in sample_episode(pi, rng.integers(0, nS), H, rng):
            V[s] += alpha * (r + G * V[s2] - V[s])    # time-limit truncation: always bootstrap
    return V

# ---------- (d) Q-learning (off-policy) with epsilon-greedy behaviour
def q_learning(n_steps, rng, eps=0.2, H=50, Qstar=None, log_every=None):
    Q = np.zeros((nS, nA)); C = np.zeros((nS, nA)); log = []
    s = rng.integers(0, nS); t_ep = 0
    for t in range(n_steps):
        a = rng.integers(0, nA) if rng.random() < eps else int(np.argmax(Q[s]))
        s2, r = E.step(s, a, rng)
        C[s, a] += 1; alpha = 1.0 / C[s, a] ** 0.6
        Q[s, a] += alpha * (r + G * Q[s2].max() - Q[s, a])
        s = s2; t_ep += 1
        if t_ep == H: s = rng.integers(0, nS); t_ep = 0         # restart (exploring starts)
        if log_every and (t + 1) % log_every == 0:
            log.append((t + 1, np.abs(Q - Qstar).max(),
                        (policy_eval_exact(Q.argmax(1)) - policy_eval_exact(Qstar.argmax(1))).min()))
    return Q, log

if __name__ == "__main__":
    t0 = time.time()
    print("temperatures (K):", E.TEMPS.astype(int).tolist(), " gamma =", G, " N =", E.N)
    for T in E.TEMPS:
        ka, kg = E.rates(T)
        print(f"  T={T:6.0f} K  k_ann={ka:.3f}  k_gen={kg:.4f}  equilibrium n = {E.N*kg/(ka+kg):.2f}")
    Vs, hist = value_iteration()
    Qs = q_from_v(Vs); pis = Qs.argmax(1)
    ratios = hist[1:] / hist[:-1]
    print(f"\n(a) value iteration: {len(hist)} sweeps to |dV|<1e-12; "
          f"successive ratio -> {ratios[50]:.4f} (k=50), {ratios[200]:.4f} (k=200); gamma = {G}")
    print("    optimal T*(n) for n=0..20:", E.TEMPS[pis].astype(int).tolist())
    print("    V*(n) at n=0,5,10,20:", np.round(Vs[[0, 5, 10, 20]], 3).tolist())
    pi_pi, V_pi, it = policy_iteration()
    print(f"(b) policy iteration: {it} evaluations; same policy as VI: {np.array_equal(pi_pi, pis)}; "
          f"max|V_PI - V_VI| = {np.abs(V_pi - Vs).max():.1e}")
    # fixed schedules for comparison, starting from n = 20
    for name, a in [("always 1100 K", 3), ("always 900 K", 2), ("always 700 K", 1)]:
        Vf = policy_eval_exact(np.full(nS, a))
        print(f"    fixed {name:13s}: V(20) = {Vf[20]:.3f}   (optimal {Vs[20]:.3f})")

    # (c) MC vs TD(0) on the optimal policy, same episodes (H=150), 20 seeds
    print("\n(c) policy evaluation of pi*: episodes of H=150 steps, random start, 20 seeds")
    Vp = policy_eval_exact(pis); res = {}
    for n_ep in (20, 200):
        for name, f in [("MC", lambda rng: mc_eval(pis, n_ep, 150, rng)),
                        ("TD0 a=0.1", lambda rng: td0_eval(pis, n_ep, 150, rng, 0.1)),
                        ("TD0 a=0.3", lambda rng: td0_eval(pis, n_ep, 150, rng, 0.3))]:
            v = np.array([f(np.random.default_rng(1000 + sd)) - Vp for sd in range(20)])
            res[(n_ep, name)] = v
            print(f"    {n_ep:4d} episodes  {name:10s}: RMSE = {np.sqrt((v**2).mean()):.3f}, "
                  f"bias = {np.abs(v.mean(0)).mean():.3f}, sd across seeds = {v.std(0).mean():.3f}")

    # (d) Q-learning vs DP ground truth
    print("\n(d) Q-learning, eps=0.2 behaviour, alpha=1/N(s,a)^0.6, 5 seeds")
    srt = np.sort(Qs, 1); gap = srt[:, -1] - srt[:, -2]
    print(f"    action gap Q*(best)-Q*(2nd): smallest {gap.min():.4f} at n={gap.argmin()}, "
          f"median {np.median(gap):.4f}")
    logs = []
    for sd in range(5):
        Q, log = q_learning(400_000, np.random.default_rng(sd), Qstar=Qs, log_every=20_000)
        logs.append(log)
        match = (Q.argmax(1) == pis).mean()
        print(f"    seed {sd}: max|Q-Q*| = {np.abs(Q-Qs).max():.3f}, greedy policy matches pi* in "
              f"{int(match*nS)}/{nS} states, worst-state value loss = {max(0.0, -log[-1][2]):.4f}, "
              f"mismatched n = {np.where(Q.argmax(1) != pis)[0].tolist()}")
    np.savez("../ch08/data/81_anneal.npz", hist=hist, Vs=Vs, pis=pis,
             logs=np.array(logs), **{f"pe_{k[0]}_{k[1].replace(' ','_').replace('=','')}": v for k, v in res.items()})
    # greedy-policy annealing trajectories from n=20 (for the figure)
    rng = np.random.default_rng(7); trajs = []
    for k in range(200):
        s = 20; tr = []
        for t in range(120):
            a = pis[s]; tr.append((s, E.TEMPS[a])); s, _ = E.step(s, a, rng)
        trajs.append(tr)
    trajs = np.array(trajs)
    np.save("../ch08/data/81_anneal_traj.npy", trajs)
    print("\nunder pi* from n=20 (200 runs): step, mean n, mean T:", [(t+1, round(float(trajs[:,t,0].mean()),2), int(round(float(trajs[:,t,1].mean())))) for t in (9,19,39,59,79,119)])
    print(f"runtime {time.time()-t0:.1f} s")
