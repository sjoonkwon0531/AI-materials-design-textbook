"""Exercises 9.4.1 - 9.4.5: closed-form numbers, each checked numerically."""
import numpy as np
from scipy import stats
D = "../../../ch09/data/"

print("=== 9.4.1 ===")
cs, c32, ci, ce, Cd, Ct = 18.261e-3, 0.321e-3, 0.125e-3, 0.628e-3, 38.0, 58.0
print(f"(a) N* = (38+58)/(18.261-0.125)ms = {(Cd + Ct) / (cs - ci):,.0f};  speed-up {cs / ci:.1f}")
cost = np.load(D + "94d_uq.npz")["cost"]; sol = np.load(D + "94d_uq.npz")["sol"]
print(f"    unrounded: data {cost[3]:.3f} s, train {cost[4]:.3f} s, c_i {cost[0]:.6f} ms, c_s {sol[0, 2]:.4f} ms -> "
      f"N* = {1e3 * (cost[3] + cost[4]) / (sol[0, 2] - cost[0]):,.0f}")
print(f"(b) ensemble: N* = (38+4*58)/(18.261-0.628)ms = {(Cd + 4 * Ct) / (cs - ce):,.0f};  unrounded "
      f"{1e3 * (cost[3] + 4 * cost[4]) / (sol[0, 2] - cost[1]):,.0f};  speed-up {cs / ce:.1f}")
print(f"(c) 32-point solver: N* = 96/(0.321-0.125)ms = {(Cd + Ct) / (c32 - ci):,.0f}; unrounded {1e3 * (cost[3] + cost[4]) / (sol[3, 2] - cost[0]):,.0f};"
      f" ensemble c_i - c_s = {1e3 * (ce - c32):.3f} ms > 0")
K, Cft = 64, 8.0
print(f"(d) few-shot: N* = (64*18.261ms + 8)/(18.261-0.125)ms = {(K * cs + Cft) / (cs - ci):,.0f}  (label cost {K * cs:.2f} s);"
      f" ratio to (a): {(Cd + Ct) / (K * cs + Cft):.1f}")
print(f"    without fine-tuning cost: {K * cs / (cs - ci):.1f} > K = 64")

print("=== 9.4.2 ===")
x = np.linspace(0, 2 * np.pi, 4096, endpoint=False); nrm = lambda f: np.sqrt((f ** 2).mean())
up, um = np.sin(x - 1), np.sin(x + 1)
mean0 = 0.5 * (up + um)
print(f"(a) max |mean - cos1 sin x| = {np.abs(mean0 - np.cos(1) * np.sin(x)).max():.1e}; cos 1 = {np.cos(1):.4f};"
      f" rel. L2 error vs sin(x-1): {nrm(mean0 - up) / nrm(up):.4f}, vs sin(x+1): {nrm(mean0 - um) / nrm(um):.4f}; sin 1 = {np.sin(1):.4f}")
y, sig = -0.6, 0.5; a = np.sin(1)
lp, lm = -(y + a) ** 2 / (2 * sig ** 2), -(y - a) ** 2 / (2 * sig ** 2)
p = 1 / (1 + np.exp(lm - lp))
print(f"(b) u(0,1) = -sin1 = {-a:.4f} (c=+1), +{a:.4f} (c=-1); log-likelihoods {lp:.4f}, {lm:.4f}; log odds {lp - lm:.4f} (= -2 y sin1 / sigma^2 = {-2 * y * a / sig**2:.4f});"
      f" p(c=+1|y) = {p:.4f}")
mean1 = p * up + (1 - p) * um
print(f"(c) rel. L2 error of the posterior mean vs sin(x-1): {nrm(mean1 - up) / nrm(up):.4f}  (2(1-p) sin1 = {2 * (1 - p) * a:.4f});"
      f" amplitude of the mean: {np.sqrt(2) * nrm(mean1):.4f} (K=0: {np.sqrt(2) * nrm(mean0):.4f})")
for pp, lab in ((0.5, "K=0"), (p, "after y")):
    var = pp * (1 - pp) * (2 * a) ** 2
    print(f"(d) {lab}: posterior-expected squared rel. error: mean {var:.4f} (rms {np.sqrt(var):.3f}), posterior sample {2 * var:.4f} (rms {np.sqrt(2 * var):.3f});"
          f" wrong-branch error 2 sin1 = {2 * a:.3f} with prob. {1 - pp:.4f} if c=+1 is true")
# Lab 9.12 analogue: |E exp(-i c k)| for c uniform in a window of half-width w
for wdt in (0.2, 0.4):
    print(f"    |E e^(-ick)| for c uniform in +-{wdt}, k = 3: {np.sin(3 * wdt) / (3 * wdt):.3f}")

print("=== 9.4.3 ===")
ratio = lambda lam: (np.exp(lam) - 1) / lam
for rho, nu, k in ((0, 0.1, 1), (3, 0.1, 1), (0, 0.1, 8)):
    lam = rho - nu * k ** 2
    # numerical check: integrate a' = lam a + 1 with RK4
    a_, n = 0.0, 20000; h = 1 / n
    for _ in range(n):
        k1 = lam * a_ + 1; k2 = lam * (a_ + h / 2 * k1) + 1; k3 = lam * (a_ + h / 2 * k2) + 1; k4 = lam * (a_ + h * k3) + 1
        a_ += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Lam = rho
    bound = np.exp(max(Lam, 0)); tight = ratio(Lam) if Lam != 0 else 1.0
    print(f"(c) (rho, nu, k) = ({rho}, {nu}, {k}): lambda = {lam:+.1f}, ||e||/||r|| = {ratio(lam):.4f} (RK4 {a_:.4f});"
          f" (e) bound e^max(Lam,0) = {bound:.3f} -> {bound / ratio(lam):.2f}x;  first form of (9.22) with Lambda = rho: {tight:.3f} -> {tight / ratio(lam):.2f}x")

print("=== 9.4.4 ===")
n, alpha = 300, 0.1
k = int(np.ceil((n + 1) * (1 - alpha))); a_, b_ = k, n + 1 - k
B = stats.beta(a_, b_)
print(f"(a) k = {k}; Beta({a_}, {b_}): mean {B.mean():.4f}, sd {B.std():.4f}; 5-95% [{B.ppf(0.05):.3f}, {B.ppf(0.95):.3f}]")
m = 300; sb = np.sqrt(0.9 * 0.1 / m); tot = np.sqrt(B.var() + sb ** 2)
bb = np.sqrt(a_ * b_ * (a_ + b_ + m) / (m * (a_ + b_) ** 2 * (a_ + b_ + 1)))
print(f"(b) binomial sd {sb:.4f}; combined {tot:.4f} (exact beta-binomial sd {bb:.4f}); (0.90 - 0.86)/sd = {(0.90 - 0.86) / tot:.2f};"
      f" from the mean {B.mean():.4f}: {(B.mean() - 0.86) / bb:.2f}")
BB = stats.betabinom(m, a_, b_)
print(f"    P(coverage <= 258/300) = {BB.cdf(258):.3f};  check 0.86*300 = {0.86 * 300:.1f}")
nn = 300
while True:
    kk = int(np.ceil((nn + 1) * 0.9)); s = stats.beta(kk, nn + 1 - kk).std()
    if s <= 0.005: break
    nn += 1
print(f"(c) smallest n with sd <= 0.005: n = {nn} (sd {s:.5f}); approx 0.09/0.005^2 - 2 = {0.09 / 0.005**2 - 2:.0f}; solver cost {nn * 18.261e-3:.1f} CPU-s"
      f" ({nn * 18.261e-3 / 38:.2f} x the training-data cost); test-side binomial sd stays {sb:.4f} unless more test points")
print(f"(d) low-nu: (0.90 - 0.207)/{tot:.4f} = {(0.90 - 0.207) / tot:.1f} sd")
uq = np.load(D + "94d_uq.npz"); q = uq["q"][3]
for s_ in ("cal", "ID", "low-nu", "high-k", "reaction"):
    r = uq[f"{s_}|err"] / uq[f"{s_}|res"]
    print(f"    {s_:9s}: median err/res {np.median(r):.3f}, share with err/res <= q = {q:.3f}: {100 * (r <= q).mean():.1f}%")

print("=== 9.4.5 ===")
e1, e4, M = 0.72, 0.56, 4
v = (e1 ** 2 - e4 ** 2) / (1 - 1 / M); b2 = e1 ** 2 - v
print(f"(b) v = {v:.4f} -> sqrt v = {np.sqrt(v):.3f} %;  b^2 = {b2:.4f} -> b = {np.sqrt(b2):.3f} %;  variance share of the single-model MSE: {v / e1**2:.2f}")
print(f"(c) M -> inf: {np.sqrt(b2):.3f} %;  M = 8: {np.sqrt(b2 + v / 8):.3f} %, M = 16: {np.sqrt(b2 + v / 16):.3f} %")
print(f"(d) ratio err/sd for the ensemble mean = sqrt(b^2 + v/M)/sqrt(v): ID model {np.sqrt(b2 + v / M) / np.sqrt(v):.2f} (Lab median 0.8)")
print(f"    reaction: ratio 186 with M = 4 -> b/sqrt(v) = sqrt(186^2 - 1/4) = {np.sqrt(186.3**2 - 0.25):.1f}; variance share of the MSE = {1 / (186.3**2 + 0.75):.1e}")
# Monte Carlo check of (a)
rng = np.random.default_rng(0); b, sv = 0.5, 0.5
xi = sv * rng.standard_normal((400000, 4)); e = b + xi
print(f"(a) Monte Carlo (b = sqrt v = 0.5): single MSE {np.mean(e[:, 0]**2):.4f} (b^2+v = {b*b+sv*sv:.4f}); mean-of-4 MSE {np.mean(e.mean(1)**2):.4f} (b^2+v/4 = {b*b+sv*sv/4:.4f});"
      f" E[sample var] = {np.mean(e.var(1, ddof=1)):.4f}")
