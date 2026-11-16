import numpy as np, warnings, time; warnings.filterwarnings("ignore")
from common import load_nonmetal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
t0=time.time()
d,X,y,names,chemsys,nel=load_nonmetal(); names=np.array(names); n,p=X.shape
lab13=["avg_dev NsValence","avg_dev Electronegativity","maximum NdValence","avg_dev NpValence","avg_dev Row","mean Row",
"range GSmagmom","range NdUnfilled","avg_dev GSvolume_pa","maximum GSmagmom","mean Electronegativity","mean NdValence","mean CovalentRadius"]
AL=[0.05,0.023,0.01]
def stab(Xm,nm,B=100,seed=0,label=""):
    rng=np.random.default_rng(seed); pp=Xm.shape[1]
    sel=np.zeros(pp); per=np.zeros((3,pp)); nnz=np.zeros((B,3))
    for b in range(B):
        ib=rng.choice(n,n//2,replace=False); Zb=StandardScaler().fit_transform(Xm[ib])
        m=Lasso(alpha=AL[0],max_iter=5000,warm_start=True); u=np.zeros(pp,bool)
        for k,a in enumerate(AL):
            m.set_params(alpha=a); m.fit(Zb,y[ib]); nz=m.coef_!=0; u|=nz; per[k]+=nz; nnz[b,k]=nz.sum()
        sel+=u
    sel/=B; per/=B
    print(f"--- {label}: p={pp}, B={B} half-subsamples; mean #nonzero at alpha 0.05/0.023/0.01 = "+"/".join(f"{v:.1f}" for v in nnz.mean(0)))
    hist=np.histogram(sel,bins=[0,0.0001,0.2,0.5,0.8,1.0001])[0]
    print("    freq histogram  =0: %d, (0,0.2): %d, [0.2,0.5): %d, [0.5,0.8): %d, >=0.8: %d"%tuple(hist))
    for k,a in enumerate(AL): print(f"    single alpha={a}: #features with freq>=0.8 = {(per[k]>=0.8).sum()}")
    top=[(nm[j],sel[j]) for j in np.argsort(-sel) if sel[j]>=0.8]
    return sel,per,top
sel,per,top=stab(X,names,label="(a) all 123 features")
S=set(t for t,_ in top)
print("    stable (>=0.8):",", ".join(f"{t} ({f:.2f})" for t,f in top))
print("    Lab13 in stable set: %d / 13; missing: %s"%(len(S&set(lab13)), [x for x in lab13 if x not in S]))
print("    stable but not in Lab13:",len(S-set(lab13)))
j=list(names).index("maximum Electronegativity"); print("    maximum Electronegativity union freq %.2f; per alpha %s"%(sel[j],np.round(per[:,j],2)))
# (b) merge |corr|>=0.95 pairs (keep the first in column order)
C=np.corrcoef(StandardScaler().fit_transform(X).T)
drop=set(); groups={}
for i in range(p):
    if i in drop: continue
    for k in range(i+1,p):
        if k not in drop and abs(C[i,k])>=0.95: drop.add(k); groups.setdefault(i,[]).append(k)
keep=[i for i in range(p) if i not in drop]
print(f"(b) pairs |corr|>=0.95: {int((np.abs(C[np.triu_indices(p,1)])>=0.95).sum())}; dropped {len(drop)} -> p={len(keep)}")
for i,ks in groups.items(): print("    keep",names[i],"<- drop",[names[k] for k in ks])
sel2,per2,top2=stab(X[:,keep],names[keep],label="(b) decorrelated")
print("    stable (>=0.8):",", ".join(f"{t} ({f:.2f})" for t,f in top2))
# compare frequency of kept representatives before/after
print("    representatives whose partner was removed: freq before -> after")
for i,ks in groups.items():
    jj=keep.index(i); print(f"      {names[i]:28s} {sel[i]:.2f} -> {sel2[jj]:.2f}   (dropped partner(s) before: {[round(sel[k],2) for k in ks]})")
print("time %.1f"%(time.time()-t0))
