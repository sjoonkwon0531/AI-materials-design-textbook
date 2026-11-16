# Cross-check of lab_1_5.jl: uses V (not Vt), 1-based truncation mirrored as i < t
import numpy as np
yr  = np.array([1930,1940,1949,1960,1970,1980,1990,2000,2010], float)
pop = np.array([2044,2355,2017,2499,3144,3741,4339,4599,4799], float)
A = np.column_stack([yr**2, yr, np.ones_like(yr)]); B = pop
U, s, Vt = np.linalg.svd(A, full_matrices=False); V = Vt.T
print("singular values: " + ", ".join(f"{si:.4g}" for si in s))
print(f"cond(A) = s1/s_min: {s[0]/s[-1]:.3g}")
for t in (3, 2, 1):
    sinv = np.array([1/s[i-1] if i <= t else 0.0 for i in range(1, 4)])
    X = V @ np.diag(sinv) @ U.T @ B
    rms = np.sqrt(np.mean((A @ X - B)**2))
    print(f"t={t}: a={X[0]:.4g} b={X[1]:.4g} c={X[2]:.4g}  RMS={rms:.2f}")
