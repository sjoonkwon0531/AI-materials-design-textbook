import numpy as np, math
from numba import njit
from scipy.spatial import cKDTree
@njit(cache=True)
def eone(x,i,L,a,xi):
    e=0.0
    for j in range(x.shape[0]):
        if j==i: continue
        d0=xi[0]-x[j,0]; d1=xi[1]-x[j,1]; d2=xi[2]-x[j,2]
        d0-=L*round(d0/L); d1-=L*round(d1/L); d2-=L*round(d2/L)
        r2=d0*d0+d1*d1+d2*d2
        if r2<1.0:
            r=math.sqrt(r2); e+=0.5*a*(1-r)**2
    return e
@njit(cache=True)
def mc(x,L,a,nsw,dmax,seed):
    np.random.seed(seed); n=x.shape[0]; acc=0
    for s in range(nsw*n):
        i=np.random.randint(n); xi=x[i].copy(); xn=xi+dmax*(np.random.random(3)-0.5)
        dE=eone(x,i,L,a,xn)-eone(x,i,L,a,xi)
        if dE<=0 or np.random.random()<math.exp(-dE):
            x[i]=xn%L; acc+=1
    return acc/(nsw*n)
L=6.0; n=648; a=25.0
rng=np.random.default_rng(0); x=rng.random((n,3))*L
mc(x,L,a,200,0.6,1)
ps=[]
for k in range(60):
    mc(x,L,a,20,0.6,k+2)
    t=cKDTree(x,boxsize=L); pr=t.query_pairs(1.0,output_type='ndarray')
    d=x[pr[:,0]]-x[pr[:,1]]; d-=L*np.round(d/L); r=np.linalg.norm(d,axis=1)
    vir=(a*(1-r)*r).sum()
    ps.append(n/L**3+vir/(3*L**3))
ps=np.array(ps); print(ps.mean(), ps.std()/np.sqrt(len(ps)))
