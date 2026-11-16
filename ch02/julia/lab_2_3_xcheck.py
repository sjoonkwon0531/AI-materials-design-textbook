# numpy mirror of lab_2_3.jl (BDF2 part + FE part); RK45 part is checked in rk45_scipy_xcheck.py
import numpy as np
from scipy.linalg import expm, lu_factor, lu_solve
A=np.array([[-1001.5,0,0],[1001.5,-0.5,1.0],[0,0.5,-1.0]]);x0=np.array([1.,0,0]);T=8.0
for dt in [0.0019,0.0020,0.0021,0.0022]:
    x=x0.copy()
    with np.errstate(all='ignore'):
        for _ in range(int(round(T/dt))): x=x+dt*(A@x)
    print("FE",dt,"%.3e"%np.abs(x).max())
prev=None
for h in [0.1,0.04,0.02,0.01]:
    n=int(round(T/h));xm=x0.copy();x=np.linalg.solve(np.eye(3)-h*A,x0);F=lu_factor(np.eye(3)-2*h/3*A)
    for _ in range(2,n+1): xn=lu_solve(F,(4*x-xm)/3);xm=x;x=xn
    e=np.abs(x-expm(A*T)@x0).max();print("BDF2",h,n,"%.2e"%e,"" if prev is None else "%.2f"%(prev/e));prev=e
