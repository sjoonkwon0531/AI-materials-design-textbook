# numpy mirror of lab_2_1.jl (identical algorithm; round(Int,T/h) == round(T/h))
import numpy as np
from scipy.linalg import expm
A=np.array([[-1.0,0.5,0.0],[1.0,-1.5,0.3],[0.0,1.0,-0.3]]); x0=np.array([1.,0,0]); T=5.0
xe=expm(A*T)@x0
fe=lambda x,h,A: x+h*(A@x); be=lambda x,h,A: np.linalg.solve(np.eye(3)-h*A,x)
def rk4(x,h,A):
    k1=A@x;k2=A@(x+h/2*k1);k3=A@(x+h/2*k2);k4=A@(x+h*k3); return x+h/6*(k1+2*k2+2*k3+k4)
for n,s in [('FE',fe),('BE',be),('RK4',rk4)]:
    for h in [0.1,0.05,0.025,0.0125]:
        x=x0.copy()
        for _ in range(int(round(T/h))): x=s(x,h,A)
        print(n,h,"%.3e"%np.linalg.norm(x-xe))
As=np.array([[-1000.0,0.5,0.0],[1000.0,-1.5,0.3],[0.0,1.0,-0.3]])
for h in [0.0019,0.0021]:
    x=x0.copy()
    for _ in range(int(round(T/h))): x=x+h*(As@x)
    print(h,"%.3e"%np.abs(x).max())
