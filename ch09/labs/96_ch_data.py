"""Data for the Case Study of 9.6: Cahn-Hilliard trajectories from the Ch.2 solver (Box 2.22).
Grid 64 x 64, L = 2 pi, eps = 0.1, dt = 0.05, S = 2 (same dx-to-eps ratio would need 128^2; see text).
Each trajectory: quench c0 = m + 0.1 * U(-1, 1), recorded every Dt = 2.5 (50 solver steps) up to T = 50.
  train : 160 trajectories, m ~ U(-0.2, 0.2), seed 0
  test  :  40 trajectories, m ~ U(-0.2, 0.2), seed 1
  ood   :  16 trajectories for each |m| in {0.3, 0.4, 0.5} (8 with +, 8 with -), seed 2
Also: (i) time-step convergence of the reference solver, (ii) the amplification factor of the scheme.
Output: ../data/96_ch_64.npz (float32)."""
import time, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("ch", "96_ch_solver.py")
ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)

N, L, EPS, DT, S, STEPS, NREC = 64, 2 * np.pi, 0.10, 0.05, 2.0, 50, 20
sol = ch.CH2D(N=N, L=L, eps=EPS, dt=DT, S=S)
t0 = time.perf_counter(); c0 = time.process_time()
rng = np.random.default_rng(0); m_tr = rng.uniform(-0.2, 0.2, 160)
tr = sol.trajectory(ch.random_ic(rng, 160, N, m_tr), NREC, STEPS)
t_train = time.perf_counter() - t0
rng = np.random.default_rng(1); m_te = rng.uniform(-0.2, 0.2, 40)
te = sol.trajectory(ch.random_ic(rng, 40, N, m_te), NREC, STEPS)
rng = np.random.default_rng(2); m_ood = np.repeat([0.3, 0.4, 0.5], 16) * np.tile(np.repeat([1.0, -1.0], 8), 3)
ood = sol.trajectory(ch.random_ic(rng, 48, N, m_ood), NREC, STEPS)
el, cpu = time.perf_counter() - t0, time.process_time() - c0
np.savez_compressed("../data/96_ch_64.npz", train=tr.astype(np.float32), test=te.astype(np.float32),
                    ood=ood.astype(np.float32), m_train=m_tr, m_test=m_te, m_ood=m_ood,
                    L=L, eps=EPS, dt=DT, S=S, steps=STEPS)
print(f"grid {N}^2, L = 2 pi, dx = {L / N:.4f}, eps = {EPS}, dt = {DT}, S = {S}; record every {STEPS} steps (Dt = {STEPS * DT})")
print(f"all 248 trajectories ({248 * NREC * STEPS:,d} solver steps): wall {el:.1f} s, CPU {cpu:.1f} s"
      f" = {1e3 * cpu / 248 / (NREC * STEPS):.3f} ms CPU per solver step per sample")
t = np.arange(NREC + 1) * STEPS * DT

# structure checks on the test set, same three checks as Lab 2.14
drift = np.abs(te.mean((2, 3)) - te[:, :1].mean((2, 3))).max()
F = np.stack([sol.free_energy(te[:, i]) for i in range(NREC + 1)], 1)
Lt = np.stack([sol.length(te[:, i]) for i in range(NREC + 1)], 1)
fit = (t >= 5) & (t <= 50)
slope = np.polyfit(np.log(t[fit]), np.log(Lt.mean(0)[fit]), 1)[0]
slopes = np.array([np.polyfit(np.log(t[fit]), np.log(l[fit]), 1)[0] for l in Lt])
print(f"[1] mass: max |mean(t) - mean(0)| over 40 test trajectories = {drift:.2e}")
print(f"[2] free energy: mean F {F[:, 0].mean():.2f} -> {F[:, -1].mean():.2f}; increases between records: {(np.diff(F, axis=1) > 0).sum()} / {F.shape[0] * NREC}")
print(f"[3] L(t): {Lt.mean(0)[1]:.2f} (t=2.5) -> {Lt.mean(0)[-1]:.2f} (t=50); slope of mean L on 5 <= t <= 50: {slope:.3f};"
      f" per-trajectory slopes {slopes.mean():.3f} +/- {slopes.std():.3f} (s.d.)")
pers = ch_err = np.sqrt(((te[:, 1:] - te[:, :-1]) ** 2).sum((2, 3)) / (te[:, 1:] ** 2).sum((2, 3))).mean(0)
print("persistence error of one big step starting at t = 0, 2.5, 5, 10, 25, 47.5:",
      " ".join(f"{pers[i]:.3f}" for i in (0, 1, 2, 4, 10, 19)))

# (i) how converged in time is the reference?  same initial conditions, smaller dt
print("\ntime-step convergence of the solver (8 test initial conditions, vs dt = 0.0025):")
c0 = te[:8, 0].astype(float)
ref = ch.CH2D(N=N, L=L, eps=EPS, dt=0.0025, S=S).trajectory(c0, NREC, 1000)
rel = lambda a, b: np.sqrt(((a - b) ** 2).sum((-2, -1)) / (b**2).sum((-2, -1))).mean()
print("   dt      rel. L2 at t=2.5   t=10     t=50     L(10)    L(50)")
print(f"   0.0025  (reference)                           {sol.length(ref[:, 4]).mean():.3f}    {sol.length(ref[:, -1]).mean():.3f}")
conv = []
for dt in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5):
    a = ch.CH2D(N=N, L=L, eps=EPS, dt=dt, S=S).trajectory(c0, NREC, int(round(2.5 / dt)))
    row = (dt, rel(a[:, 1], ref[:, 1]), rel(a[:, 4], ref[:, 4]), rel(a[:, -1], ref[:, -1]),
           sol.length(a[:, 4]).mean(), sol.length(a[:, -1]).mean())
    conv.append(row)
    print(f"   {dt:<6g}  {row[1]:.3f}              {row[2]:.3f}    {row[3]:.3f}    {row[4]:.3f}    {row[5]:.3f}")

# (i-b) the same question after the pattern has formed: advance the true c(5) by one big step
print("one big step (Dt = 2.5) from the formed microstructure c(5) and c(25), 40 test trajectories, vs dt = 0.0025:")
r5 = ch.CH2D(N=N, L=L, eps=EPS, dt=0.0025, S=S).advance(te[:, 2].astype(float), 1000)
r25 = ch.CH2D(N=N, L=L, eps=EPS, dt=0.0025, S=S).advance(te[:, 10].astype(float), 1000)
conv2 = []
for dt in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 2.5):
    s2 = ch.CH2D(N=N, L=L, eps=EPS, dt=dt, S=S); n = int(round(2.5 / dt))
    conv2.append((dt, rel(s2.advance(te[:, 2].astype(float), n), r5), rel(s2.advance(te[:, 10].astype(float), n), r25)))
    print(f"   dt = {dt:<6g}: rel. L2 from c(5): {conv2[-1][1]:.4f}   from c(25): {conv2[-1][2]:.4f}")

# (ii) von Neumann amplification factor of the scheme around a uniform state c = m
print("\namplification of the fastest mode per unit time, scheme (dt = 0.05) vs exact exp(sigma):")
k = np.arange(1, 33).astype(float)
for m in (0.0, 0.2, 0.3, 0.4, 0.5):
    a = 1 - 3 * m**2
    g = (1 + DT * k**2 * (S + a)) / (1 + DT * S * k**2 + DT * EPS**2 * k**4)
    sig = k**2 * (a - EPS**2 * k**2)
    onset = np.log(1.0 / 0.0577) / (np.log(g.max()) / DT)      # time for the rms noise 0.0577 to reach O(1)
    print(f"   m = {m:.1f}: scheme  max g^(1/dt) = {g.max() ** (1 / DT):8.2f} at k = {k[g.argmax()]:.0f};"
          f"  PDE  exp(sigma_max) = {np.exp(sig.max()):10.3g} at k = {k[sig.argmax()]:.0f};"
          f"  time for 0.058 -> 1 with the scheme: {onset:.2f}")
std_ood = (ood - ood.mean((2, 3), keepdims=True)).std((2, 3))
for m in (0.3, 0.4, 0.5):
    s = std_ood[np.abs(m_ood) == m].mean(0)
    print(f"   |m| = {m}: std of c - m at t = 2.5, 5, 7.5, 10: " + " ".join(f"{s[i]:.3f}" for i in (1, 2, 3, 4)))
s = (te - te.mean((2, 3), keepdims=True)).std((2, 3)).mean(0)
print("   |m| <= 0.2 (test): std of c - m at t = 2.5, 5, 7.5, 10: " + " ".join(f"{s[i]:.3f}" for i in (1, 2, 3, 4)))
np.savez("../data/96_ch_data_checks.npz", conv=np.array(conv), conv2=np.array(conv2), t=t, F=F, Lt=Lt)
