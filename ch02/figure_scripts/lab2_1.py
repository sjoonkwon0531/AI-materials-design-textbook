"""Lab 2.1: explicit Euler / implicit Euler / RK4 vs expm baseline (code block b00 completed)."""
import numpy as np
from scipy.linalg import expm
A = np.array([[-1.0, 0.5, 0.0],[1.0,-1.5,0.3],[0.0,1.0,-0.3]])
x0 = np.array([1.0,0.0,0.0]); T=5.0
print("col sums",A.sum(0),"eig",np.sort(np.linalg.eigvals(A).real))
exact = lambda t: expm(A*t)@x0
def march(step,h,A=A):
    n=int(round(T/h)); x=x0.copy()
    for _ in range(n): x=step(x,h,A)
    return x
fe=lambda x,h,A: x+h*(A@x)
be=lambda x,h,A: np.linalg.solve(np.eye(3)-h*A,x)
def rk4(x,h,A):
    k1=A@x;k2=A@(x+h/2*k1);k3=A@(x+h/2*k2);k4=A@(x+h*k3); return x+h/6*(k1+2*k2+2*k3+k4)
xe=exact(T)
for name,s in [('FE',fe),('BE',be),('RK4',rk4)]:
    prev=None
    for h in [0.1,0.05,0.025,0.0125]:
        e=np.linalg.norm(march(s,h)-xe)
        print(name,h,"%.3e"%e, "" if prev is None else "ratio %.2f"%(prev/e)); prev=e
# stiffen: scale one rate constant (k21: 1->1000)
As=np.array([[-1000.0,0.5,0.0],[1000.0,-1.5,0.3],[0.0,1.0,-0.3]])
lam=np.linalg.eigvals(As); print("stiff eig",np.sort(lam.real)); lmax=abs(lam.real).max()
print("2/|lmax|=",2/lmax)
for h in [0.0019,0.0021]:
    n=int(round(T/h)); x=x0.copy()
    for _ in range(n): x=x+h*(As@x)
    print("FE stiff h=",h,"|x|=%.3e"%np.abs(x).max())
