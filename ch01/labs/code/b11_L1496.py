import numpy as np
yr  = np.array([1930,1940,1949,1960,1970,1980,1990,2000,2010], float)
pop = np.array([2044,2355,2017,2499,3144,3741,4339,4599,4799], float)
A = np.column_stack([yr**2, yr, np.ones_like(yr)]); B = pop
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", s)          # [1.165e+07, 7.768e+01, 4.508e-04]
print("cond(A) = s0/s_min:", s[0]/s[-1])  # 2.58e+10  (!!)
def fit(t):
    sinv = np.array([1/si if i < t else 0.0 for i, si in enumerate(s)])
    X = Vt.T @ np.diag(sinv) @ U.T @ B
    rms = np.sqrt(np.mean((A @ X - B)**2))
    return X, rms
for t in (3, 2, 1):
    X, rms = fit(t); print(f"t={t}: a={X[0]:.4g} b={X[1]:.4g} c={X[2]:.4g}  RMS={rms:.2f}")