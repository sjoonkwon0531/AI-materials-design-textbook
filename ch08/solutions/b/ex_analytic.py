import numpy as np
from common import lab
g = 0.9
print("== 8.4.1")
Vs = 1 / (1 - g); Vc = lambda p: 2 / (1 - g * p)
print("V_safe", Vs, "V_chal(0.9)", Vc(0.9), "V_chal(0.8)", Vc(0.8), "V_chal(0.95)", Vc(0.95), "p* =", 0.8 / 0.9)
# robust value iteration with p in [0.8, 0.95]
V = 0.0
for _ in range(500): V = max(1 + g * V, min(2 + g * p * V for p in (0.8, 0.95)))
print("robust VI V(A) =", V, " Q_chal at robust V =", 2 + g * 0.8 * V)
print("== 8.4.4")
for e in [1]:
    for T in (30, 60, 120, 240): print(T, sum(e * (T - t + 1) for t in range(1, T + 1)), T * (T + 1) / 2)
T = np.array([30, 60, 120, 240.]); bc = np.array([323, 2048, 3020, 4740.]); dg = np.array([349, 511, 586, 644.])
for nm, y in (("BC", bc), ("DAgger", dg)):
    b, a = np.polyfit(np.log(T), np.log(y), 1)
    print(nm, "slope all 4: %.3f" % b, "pairwise", np.round(np.diff(np.log(y)) / np.diff(np.log(T)), 3),
          "slope T>=60: %.3f" % np.polyfit(np.log(T[1:]), np.log(y[1:]), 1)[0], "ratio 240/30 %.2f" % (y[-1] / y[0]))
# saturating model: excess = c * (1 - exp(-T/tau))-type? per-step excess over expert from Lab: BC (T * meanerr) ...
print("== 8.5.1")
for m in range(8, 13): q2 = 4 * np.sin(np.pi * m / 128) ** 2; print(m, round(q2, 5), q2 < 0.25)
print("m_c =", 128 / np.pi * np.arcsin(0.5 * np.sqrt(0.25)))
print("== 8.5.2")
kB = 8.617e-5
f = lambda E, T1, T0: np.exp(E / kB * (1 / T0 - 1 / T1))
print("tb x (700->740) %.2f  kappa x %.3f" % (f(3.0, 740, 700), f(1.0, 740, 700)))
print("tb x (685->725) %.2f  kappa x %.3f ; (685->645): tb %.4f kappa %.3f" % (f(3.0, 725, 685), f(1.0, 725, 685), f(3.0, 645, 685), f(1.0, 645, 685)))
E = lab("85_film_env"); P = E.FilmParams()
for dl in (-35, 0, 35):
    J, w, TB = E.rollout_batch(lambda t, o: np.full(len(o), -0.15), 2000, 5, P, delta=dl)
    print(f"const 685 K, delta {dl:+d}: J {J.mean():.3f} w_N {w.mean():.3f} lam*TB {P.lam*TB.mean():.3f}")
for Tact in (650, 685, 695, 720, 725):
    J, w, TB = E.rollout_batch(lambda t, o: np.full(len(o), (Tact - 700) / 100), 2000, 5, P, delta=0.0)
    print(f"actual {Tact} K: J {J.mean():.3f} w_N {w.mean():.3f} lam*TB {P.lam*TB.mean():.3f}")
print("== 8.6.1")
y = np.array([0.5, 0.3, 0.7, 0.6]); cm = np.maximum.accumulate(y)
print("oi", -(y[1:] - cm[:-1]).sum(), "bsf", -cm[1:].sum(), "stay oi", 0, "stay bsf", -3 * 0.5)
print("== 8.6.2")
mu, s = 1.0, 0.1
print("exact var", mu**4 / s**2 + 14 * mu**2 + 15 * s**2, "pathwise", 4 * s**2, "ratio", (mu**4 / s**2 + 14 * mu**2 + 15 * s**2) / (4 * s**2))
xi = np.random.default_rng(0).standard_normal(4_000_000); gh = -(mu + s * xi) ** 2 * xi / s
print("MC mean %.4f var %.2f" % (gh.mean(), gh.var()))
