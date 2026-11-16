import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def exact(x): return np.sin(np.pi*x)
def ffun(x):  return np.pi**2*np.sin(np.pi*x)

def fem_p1(N):                       # N elements, N-1 interior unknowns
    h = 1.0/N; x = np.linspace(0,1,N+1); nint = N-1
    # 강성행렬: (1/h)[-1, 2, -1]  ← FDM 라플라시안 * h
    K = diags([np.full(nint-1,-1/h), np.full(nint,2/h),
               np.full(nint-1,-1/h)], [-1,0,1], format='csr')
    # 하중벡터: 요소마다 2점 Gauss 구적 (§1.7 회수)
    b = np.zeros(N+1); gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    for e in range(N):
        x0 = x[e]
        for xi in gp:
            xg = x0 + 0.5*h*(1+xi)
            N0, N1, jac = 0.5*(1-xi), 0.5*(1+xi), 0.5*h
            b[e]   += ffun(xg)*N0*jac
            b[e+1] += ffun(xg)*N1*jac
    u = np.zeros(N+1); u[1:N] = spsolve(K, b[1:N])
    return x, u

def l2_err(x, u, nf=20):              # L2 error of the P1 interpolant (fine-grid)
    xf = np.linspace(0,1,(len(x)-1)*nf+1); uf = np.interp(xf, x, u)
    return np.sqrt(np.trapezoid((uf-exact(xf))**2, xf))
