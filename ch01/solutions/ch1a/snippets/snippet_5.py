import numpy as np
M = np.roll(np.eye(6), 1, axis=1); M = M + M.T          # 고리 인접행렬
x = np.sort(np.linalg.eigvalsh(M))[::-1]                 # E = alpha + x*beta
print(x, "HOMO x =", x[2], "LUMO x =", x[3], "E_pi = 6a + %gb" % (2*x[:3].sum()))
for E in (4.90, 6.20, 6.94): print(E, "eV", 1239.84/E, "nm, |beta| =", E/2)
