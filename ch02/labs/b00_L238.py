import numpy as np
from scipy.linalg import expm
A = np.array([[-1.0, 0.5, 0.0],[1.0,-1.5,0.3],[0.0,1.0,-0.3]])  # mass-conserving
x0 = np.array([1.0,0.0,0.0]); T=5.0
exact = lambda t: expm(A*t)@x0                          # eigendecomposition baseline
def march(step, h):
    n=int(T/h); x=x0.copy()
    for _ in range(n): x=step(x,h)
    return x
fe = lambda x,h: x+h*(A@x)                               # explicit Euler
be = lambda x,h: np.linalg.solve(np.eye(3)-h*A, x)       # implicit Euler (§1.4 solve)
def rk4(x,h):                                            # classical RK4 (§2.2.3)
    k1=A@x; k2=A@(x+h/2*k1); k3=A@(x+h/2*k2); k4=A@(x+h*k3)
    return x+h/6*(k1+2*k2+2*k3+k4)
