# 연습문제 1.3.A: 전진차분 오차 vs h (f=e^x, x=1) + 추가 도전(중심차분)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
e = np.exp(1.0); eps = np.finfo(float).eps
hs = 10.0 ** -np.arange(1, 16, 2)
fwd = np.abs((np.exp(1 + hs) - np.exp(1)) / hs - e)
for h, err in zip(hs, fwd):
    print(f"h=1e{int(np.log10(h)):+d}: |err| = {err:.3e}   model e*h/2 + e*eps/h = {e*h/2 + e*eps/h:.3e}")
print("best h among odd powers:", hs[fwd.argmin()], " err:", fwd.min())
# 촘촘한 스캔
hf = np.logspace(-15, -1, 1401)
ef = np.abs((np.exp(1 + hf) - np.exp(1)) / hf - e)
# 잡음이 있으므로 이동 중앙값으로 최적점 추정
from scipy.ndimage import median_filter
sm = median_filter(ef, size=41)
print(f"fine-scan optimum (median-smoothed): h ~ {hf[sm.argmin()]:.2e}, err ~ {sm.min():.2e}")
print(f"model E(h)=(e/2)h + e*eps/h: h* = sqrt(2 eps) = {np.sqrt(2*eps):.2e}, E(h*) = e*sqrt(2 eps) = {e*np.sqrt(2*eps):.2e}; sqrt(eps) = {np.sqrt(eps):.2e}")
# 추가 도전: 중심차분
ec = np.abs((np.exp(1 + hf) - np.exp(1 - hf)) / (2*hf) - e)
smc = median_filter(ec, size=41)
print(f"central diff optimum: h ~ {hf[smc.argmin()]:.2e}, err ~ {smc.min():.2e};  eps^(1/3) = {eps**(1/3):.2e}")
plt.loglog(hf, ef, lw=.8, label="forward"); plt.loglog(hf, ec, lw=.8, label="central")
plt.loglog(hs, fwd, "o"); plt.xlabel("h"); plt.ylabel("|error|"); plt.legend(); plt.savefig("ex_1_3_A.png", dpi=120)
