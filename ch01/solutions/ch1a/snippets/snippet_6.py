import numpy as np
yr = np.array([1930,1940,1949,1960,1970,1980,1990,2000,2010], float)
B = np.array([2044,2355,2017,2499,3144,3741,4339,4599,4799], float)
A = np.column_stack([yr**2, yr, np.ones_like(yr)])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
Xa = np.linalg.pinv(A) @ B
Xb = np.linalg.inv(A.T @ A) @ A.T @ B
Xc = Vt.T @ np.diag(1/s) @ U.T @ B
print(-np.log10(np.abs(Xb - Xc)/np.abs(Xc)))           # (b)가 갈라지는 자릿수
D = 1/np.linalg.norm(A, axis=0); print(np.linalg.cond(A), np.linalg.cond(A*D))
for t in (1, 2, 3):
    X = Vt.T @ np.diag([1/v if i < t else 0 for i, v in enumerate(s)]) @ U.T @ B
    print(t, np.sqrt(np.mean((A@X - B)**2)), np.linalg.norm(X))
