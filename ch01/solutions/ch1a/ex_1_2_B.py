# 연습문제 1.2.B: Antoine 식 FP32 vs FP64
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
A, B, C = 8.07131, 1730.63, 233.426
def P(T, dt):
    a, b, c = dt(A), dt(B), dt(C); T = T.astype(dt)
    return dt(10.0) ** (a - b / (c + T))
T = np.linspace(1, 100, 991)
p64 = P(T, np.float64); p32 = P(T, np.float32).astype(np.float64)
abs_d = np.abs(p32 - p64); rel_d = abs_d / p64
eps32 = np.finfo(np.float32).eps
for t in [1, 25, 50, 75, 100]:
    i = np.argmin(abs(T - t))
    print(f"T={T[i]:5.1f}C  P64={p64[i]:.6f} mmHg  |dP|={abs_d[i]:.3e}  rel={rel_d[i]:.3e}  rel/eps32={rel_d[i]/eps32:.2f}")
print(f"max abs diff {abs_d.max():.3e} at T={T[abs_d.argmax()]:.1f}C")
print(f"max rel diff {rel_d.max():.3e} at T={T[rel_d.argmax()]:.1f}C  (=  {rel_d.max()/eps32:.2f} eps32)")
lo = T < 50; hi = T >= 50
print(f"mean rel diff T<50: {rel_d[lo].mean():.3e}, T>=50: {rel_d[hi].mean():.3e}")
print(f"mean abs diff T<50: {abs_d[lo].mean():.3e}, T>=50: {abs_d[hi].mean():.3e}")
# 이론적 상대오차 스케일: ln(10)*|log10 P|*u (지수의 반올림이 증폭)
logP = np.log10(p64)
print(f"log10P range: {logP.min():.3f} .. {logP.max():.3f}")
print(f"ln10*log10P*u at T=1: {np.log(10)*abs(logP[0])*eps32/2:.2e}, at T=100: {np.log(10)*abs(logP[-1])*eps32/2:.2e}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].semilogy(T, abs_d + 1e-30, ".", ms=2); ax[0].set_xlabel("T (C)"); ax[0].set_ylabel("|P32-P64| (mmHg)")
ax[1].semilogy(T, rel_d + 1e-30, ".", ms=2); ax[1].axhline(eps32/2, ls="--"); ax[1].set_xlabel("T (C)"); ax[1].set_ylabel("relative diff")
fig.tight_layout(); fig.savefig("ex_1_2_B.png", dpi=120)
u = eps32/2
print(f"ln10*A*u (abs. rounding of A-B/(C+T) amplified) = {np.log(10)*A*u:.2e}")
print(f"cancellation factor A/log10P: T=1 -> {A/logP[0]:.1f}, T=100 -> {A/logP[-1]:.1f}")
print(f"rel diff percentiles (5,50,95): {np.percentile(rel_d,[5,50,95])}")
