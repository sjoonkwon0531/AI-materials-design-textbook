import numpy as np
from scipy.optimize import brentq
s=np.array([10,1,0.1])
for lam in (0.01,1,100):
    f=s**2/(s**2+lam); print(lam, np.round(f,6), 'df=%.4f'%f.sum())
g=lambda l: (s**2/(s**2+l)).sum()-2
l2=brentq(g,1e-6,1e3); print('df=2 at lambda=%.6f'%l2, np.round(s**2/(s**2+l2),4))
# LASSO 1D
S=lambda r,a: np.sign(r)*max(abs(r)-a,0)
print(S(0.8,0.3),S(-0.2,0.3)); print('ridge',0.8/(1+np.array([0.1,1,10])), -0.2/(1+np.array([0.1,1,10])))
# 6.4.1
C=np.array([[4,2],[2,3.]]); w,V=np.linalg.eigh(C); print(w,V, w[-1]/w.sum())
C2=np.array([[400,20],[20,3.]]); w2,V2=np.linalg.eigh(C2); print(w2,V2[:,-1], w2[-1]/w2.sum(), np.degrees(np.arctan2(V2[1,-1],V2[0,-1])), np.degrees(np.arctan2(V[1,-1],V[0,-1])))
R=C/np.sqrt(np.outer(np.diag(C),np.diag(C))); wr,Vr=np.linalg.eigh(R); print(R, wr, Vr[:,-1], wr[-1]/2)
