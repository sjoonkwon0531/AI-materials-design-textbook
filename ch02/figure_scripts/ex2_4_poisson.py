import numpy as np
from scipy.linalg import solve_banded
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
# ---- Worked Example A: 1D Poisson u''=1, u(0)=2, u(1)=2.2
ex1=lambda x: x**2/2-0.3*x+2
def poisson1d(h, method='thomas'):
    N=int(round(1/h)); x=np.linspace(0,1,N+1); n=N-1
    F=h**2*np.ones(n); F[0]-=2.0; F[-1]-=2.2
    if method=='thomas':
        ab=np.zeros((3,n)); ab[0,1:]=1; ab[1,:]=-2; ab[2,:-1]=1
        ui=solve_banded((1,1),ab,F)
    else:
        A=np.diag(-2*np.ones(n))+np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
        ui=np.linalg.solve(A,F)
    u=np.r_[2.0,ui,2.2]; return x,u
for m in ['thomas','dense']:
    for h in [0.2,0.05,0.01]:
        x,u=poisson1d(h,m); print(m,"h=%g max|u-exact| = %.1e"%(h,np.abs(u-ex1(x)).max()))
# ---- Worked Example B: 2D Poisson
ex2=lambda x,y: -np.sin(3*np.pi*x)*np.cos(5*np.pi*y)/(17*np.pi**2)
f2=lambda x,y: 2*np.sin(3*np.pi*x)*np.cos(5*np.pi*y)
def poisson2d(h,Lx=1.0,Ly=2.0,return_A=False):
    Nx=int(round(Lx/h)); Ny=int(round(Ly/h))
    x=np.linspace(0,Lx,Nx+1); y=np.linspace(0,Ly,Ny+1)
    nx,ny=Nx-1,Ny-1                       # interior counts; j (y) fastest
    X,Y=np.meshgrid(x[1:-1],y[1:-1],indexing='ij')
    D=sp.diags([np.ones(ny-1),-4*np.ones(ny),np.ones(ny-1)],[-1,0,1])
    A=sp.kron(sp.eye(nx),D)+sp.kron(sp.diags([np.ones(nx-1),np.ones(nx-1)],[-1,1]),sp.eye(ny))
    A=A.tocsr()
    F=h**2*f2(X,Y)
    ub=ex2  # Dirichlet data from exact solution
    F[0,:]-=ub(x[0],y[1:-1]); F[-1,:]-=ub(x[-1],y[1:-1])
    F[:,0]-=ub(x[1:-1],y[0]); F[:,-1]-=ub(x[1:-1],y[-1])
    U=spsolve(A,F.ravel()).reshape(nx,ny)
    err=np.abs(U-ex2(X,Y)).max()
    return (x,y,U,err,A) if return_A else err
prev=None
for h in [0.05,0.025,0.0125,0.00625]:
    e=poisson2d(h); r='' if prev is None else "(ratio %.2fx)"%(prev/e)
    print("h=%g max|u-exact| = %.2e %s"%(h,e,r)); prev=e
