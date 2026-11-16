import sys; sys.path.insert(0,'../../ch02/labs')
import numpy as np
exec(open('../../ch02/labs/b03_L1217.py').read())
def l2_err(x,u,nf=20):
    # piecewise-linear interpolant evaluated on fine sub-grid, trapezoid integral
    xf=np.linspace(0,1,(len(x)-1)*nf+1); uf=np.interp(xf,x,u)
    return np.sqrt(np.trapezoid((uf-exact(xf))**2,xf))
def l2_gauss(x,u):
    # exact-ish: 5-pt Gauss per element
    g,w=np.polynomial.legendre.leggauss(5); s=0
    for e in range(len(x)-1):
        a,b=x[e],x[e+1]; xg=0.5*(a+b)+0.5*(b-a)*g
        ug=u[e]+(u[e+1]-u[e])*(xg-a)/(b-a)
        s+=np.sum(w*(ug-exact(xg))**2)*0.5*(b-a)
    return np.sqrt(s)
print(f"{'N':>5} {'h':>8} {'max_err':>12} {'L2_err':>12} {'L2_order':>9} {'L2gauss':>12} {'nodal_order':>6}")
prev=None;pm=None
for N in [4,8,16,32,64,128,256]:
    x,u=fem_p1(N); me=np.max(np.abs(u-exact(x))); e=l2_err(x,u); eg=l2_gauss(x,u)
    o='--' if prev is None else f"{np.log2(prev/e):.3f}"
    om='--' if pm is None else f"{np.log2(pm/me):.3f}"
    print(f"{N:5d} {1/N:8.5f} {me:12.3e} {e:12.3e} {o:>9} {eg:12.3e} {om}")
    prev=e;pm=me
x,u=fem_p1(8); h=1/8
from scipy.sparse import diags
nint=7; K=diags([np.full(nint-1,-1/h), np.full(nint,2/h), np.full(nint-1,-1/h)],[-1,0,1]).toarray()
print("off=%.4f diag=%.4f 1/h=%.4f"%(K[3,2],K[3,3],1/h))
