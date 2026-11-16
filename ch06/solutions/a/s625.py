import numpy as np
from scipy.optimize import least_squares
from ase.build import molecule
def cm(Z,R):
    Z=np.asarray(Z,float); D=np.linalg.norm(R[:,None]-R[None],axis=-1); np.fill_diagonal(D,1.0)
    M=np.outer(Z,Z)/D; np.fill_diagonal(M,0.5*Z**2.4); return M
def eigs(M): return np.sort(np.linalg.eigvalsh(M))[::-1]
def pairsig(Z,R):
    out={}
    n=len(Z)
    for i in range(n):
        for j in range(i+1,n):
            k=tuple(sorted((Z[i],Z[j]))); out.setdefault(k,[]).append(np.linalg.norm(R[i]-R[j]))
    return {k:np.sort(v) for k,v in out.items()}
def sigdiff(a,b): return max(np.abs(a[k]-b[k]).max() for k in a)
def run(name,Z,base,nstart=40,seed=7):
    rng=np.random.default_rng(seed)
    Z=np.array(Z); n=len(Z)
    R0=base+rng.normal(0,0.12,base.shape)
    lam0=eigs(cm(Z,R0)); s0=pairsig(Z,R0)
    res=lambda x: eigs(cm(Z,x.reshape(n,3)))-lam0
    conv=0; same=0; diff=[]; sols=[]
    for t in range(nstart):
        x0=(base+rng.normal(0,0.35,base.shape)).ravel()
        sol=least_squares(res,x0,xtol=1e-15,ftol=1e-15,gtol=1e-15)
        if np.abs(sol.fun).max()<1e-10:
            conv+=1; R1=sol.x.reshape(n,3); d=sigdiff(s0,pairsig(Z,R1))
            if d<1e-6: same+=1
            else: diff.append(d); sols.append(R1)
    print(f"== {name}: N={n}, 3N-6={3*n-6}, starts={nstart}, converged={conv}, same-as-R0={same}, different={len(diff)}")
    print("   R0 pair distances:", {k:np.round(v,3).tolist() for k,v in s0.items()})
    if diff:
        print("   max sorted-distance difference among different solutions: min %.3f, median %.3f, max %.3f"%(min(diff),np.median(diff),max(diff)))
        # distinct solution classes
        cls=[]
        for R1 in sols:
            s1=pairsig(Z,R1)
            if not any(sigdiff(s1,c)<1e-6 for c in cls): cls.append(s1)
        print("   distinct non-equivalent solution classes:",len(cls))
        for c in cls[:4]: print("     ",{k:np.round(v,3).tolist() for k,v in c.items()})
    return R0
run("CH3-type",[6,1,1,1],molecule("CH3").positions,nstart=40)
run("H2O-type",[8,1,1],molecule("H2O").positions,nstart=40)
run("H2O-type seed 11",[8,1,1],molecule("H2O").positions,nstart=40,seed=11)
run("H2O-type seed 12",[8,1,1],molecule("H2O").positions,nstart=40,seed=12)
