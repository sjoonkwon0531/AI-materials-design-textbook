"""Ex 1.9.D post-analysis of ex1_9_D_res.npy: check whether the chi' peak lies inside the T grid."""
import numpy as np
Tc = 2/np.log(1+np.sqrt(2)); Ls = [8, 16, 32, 64]
R = np.load("ex1_9_D_res.npy")
for L, r in zip(Ls, R):
    i = np.argmax(r[:, 1]); print(f"L={L}: grid argmax T={r[i,0]:.2f} (grid {r[0,0]:.2f}-{r[-1,0]:.2f}), chi' at edge T=2.40: {r[-1,1]:.3f}, max {r[i,1]:.3f}")
Ts = {32: None, 64: None}; cm = {}
for L, r in zip(Ls, R):
    if L < 32: continue
    i = np.argmax(r[:, 1]); sl = slice(i-3, i+4); c = np.polyfit(r[sl, 0], r[sl, 1], 2)
    Ts[L] = -c[1]/(2*c[0]); cm[L] = np.polyval(c, Ts[L])
print(f"T*(32)-Tc={Ts[32]-Tc:.4f}, T*(64)-Tc={Ts[64]-Tc:.4f}, 1/nu = log2 ratio = {np.log2((Ts[32]-Tc)/(Ts[64]-Tc)):.3f}")
print(f"gamma/nu from chi'_max(32->64) = {np.log2(cm[64]/cm[32]):.3f}")
