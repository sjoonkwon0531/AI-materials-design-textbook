# numpy mirror of lab_2_8.jl incl. its own interpolation/trapezoid (vs np.interp+np.trapezoid)
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
ex=lambda x:np.sin(np.pi*x); ff=lambda x:np.pi**2*np.sin(np.pi*x)
def fem(N):
    h=1/N;x=np.linspace(0,1,N+1);n=N-1
    K=diags([np.full(n-1,-1/h),np.full(n,2/h),np.full(n-1,-1/h)],[-1,0,1],format='csr')
    b=np.zeros(N+1)
    for e in range(N):
        for xi in (-1/np.sqrt(3),1/np.sqrt(3)):
            xg=x[e]+0.5*h*(1+xi);b[e]+=ff(xg)*0.5*(1-xi)*0.5*h;b[e+1]+=ff(xg)*0.5*(1+xi)*0.5*h
    u=np.zeros(N+1);u[1:N]=spsolve(K,b[1:N]);return x,u
def l2(x,u,nf=20):
    N=len(x)-1;xf=np.linspace(0,1,N*nf+1);uf=[]
    for xs in xf:
        e=min(int(np.floor(xs*N)),N-1);t=(xs-x[e])/(x[e+1]-x[e]);uf.append((1-t)*u[e]+t*u[e+1])
    d2=(np.array(uf)-ex(xf))**2;dx=xf[1]-xf[0];return np.sqrt(dx*(d2.sum()-0.5*(d2[0]+d2[-1])))
def l2ref(x,u,nf=20):
    xf=np.linspace(0,1,(len(x)-1)*nf+1);return np.sqrt(np.trapezoid((np.interp(xf,x,u)-ex(xf))**2,xf))
p=pm=None
for N in [4,8,16,32,64,128,256]:
    x,u=fem(N);me=np.abs(u-ex(x)).max();e=l2(x,u)
    print(N,"%.3e %.3e %s %s | ref %.3e"%(me,e,'--' if p is None else "%.3f"%np.log2(p/e),'--' if pm is None else "%.3f"%np.log2(pm/me),l2ref(x,u)));p=e;pm=me
