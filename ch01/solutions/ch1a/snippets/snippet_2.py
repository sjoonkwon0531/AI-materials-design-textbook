import numpy as np, scipy.linalg as sl
from scipy.sparse.linalg import cg
for n in range(2, 17):
    H = sl.hilbert(n); b = H.sum(1); e = lambda x: np.linalg.norm(x-1)/np.sqrt(n)
    Q, Rm = np.linalg.qr(H)
    try: xc = e(sl.cho_solve(sl.cho_factor(H), b))
    except np.linalg.LinAlgError: xc = np.nan
    print(n, np.linalg.cond(H), e(sl.lu_solve(sl.lu_factor(H), b)), xc,
          e(sl.solve_triangular(Rm, Q.T@b)), e(cg(H, b, rtol=1e-15, atol=0, maxiter=20*n)[0]))
