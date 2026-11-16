import json, numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
d = json.load(open("../../../ch07/labs/76_mlp_bandgap_oof.json"))
s = d["random 5-fold (seed 0)"]; y = np.array(s["y"]); mu = np.array(s["mu"]); sd = np.array(s["sd"]); gb = np.array(s["gb"])
met = y == 0; f = met.mean(); print("n", len(y), "metals", met.sum(), f)
for name, p in (("ens", mu), ("gb", gb)):
    a, m_, n_ = [np.abs(p[k] - y[k]).mean() for k in (np.ones_like(met), met, ~met)]
    print(name, round(a, 4), round(m_, 4), round(n_, 4), "weighted", f * m_ + (1 - f) * n_)
print("(b) gb with metals 0.150:", f * 0.150 + (1 - f) * np.abs(gb[~met] - y[~met]).mean(), "rounded inputs:", 0.532*0.150+0.468*0.481)
print("check rounded ens:", 0.532*0.150+0.468*0.557)
# 7.6.2
err = np.abs(mu - y)
cov = (err <= 1.96 * sd).mean(); print("coverage", cov)
k = 1.96 / norm.ppf(0.5 + 0.730 / 2); print("k from 73.0%:", k, "z", norm.ppf(0.5+0.73/2))
k2 = 1.96 / norm.ppf(0.5 + 0.708 / 2); print("k from 70.8%:", k2)
q = np.quantile(sd, np.linspace(0, 1, 11)); b = np.clip(np.digitize(sd, q[1:-1]), 0, 9)
ms = np.array([sd[b == i].mean() for i in range(10)]); me = np.array([err[b == i].mean() for i in range(10)])
print("decile sd", np.round(ms, 4)); print("decile MAE", np.round(me, 3)); print("ratio obs/gauss", np.round(me / (np.sqrt(2/np.pi)*ms), 2))
print("pred with k:", np.round(k*np.sqrt(2/np.pi)*ms, 3))
# best single k by matching MAE overall, and coverage per decile
print("per-decile coverage of 1.96 sd:", np.round([(err[b==i] <= 1.96*sd[b==i]).mean() for i in range(10)], 2))
# fraction exactly metals predicted 0 in lowest decile
i0 = b == 0; print("lowest decile: frac metals", met[i0].mean(), "mean sd", sd[i0].mean(), "frac err==0", (err[i0]<1e-6).mean(), "frac y>0", (y[i0]>0).mean(), "MAE of nonmetals in decile", err[i0 & ~met].mean() if (i0&~met).any() else None)
# required k to explain lowest decile MAE
print("k needed lowest decile:", me[0]/(np.sqrt(2/np.pi)*ms[0]), " highest:", me[-1]/(np.sqrt(2/np.pi)*ms[-1]))
# affine calibration sigma' = a + k sigma fit on MAE of deciles
A = np.vstack([np.ones(10), np.sqrt(2/np.pi)*ms]).T; c = np.linalg.lstsq(A, me, rcond=None)[0]; print("affine fit MAE = a + k*sqrt(2/pi)*sd:", c)
