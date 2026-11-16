"""Ex 6.8.5: a miniature SISSO-like search (feature construction + 1-threshold stump) trained on one anion family."""
import numpy as np, pandas as pd, itertools, time
d=pd.read_csv("../../../ch06/labs/bartel_TableS1.csv").rename(columns={"rA (Ang)":"rA","rB (Ang)":"rB","rX (Ang)":"rX"})
y=(d.exp_label>0).astype(int).values; hal=d.X.isin(["F","Cl","Br","I"]).values; oxi=~hal
raw={k:d[k].values.astype(float) for k in ("rA","rB","rX","nA","nB","nX")}
t0=time.time()
feat={}
for k,v in raw.items(): feat[k]=v                                         # level 0
for a,b in itertools.permutations(raw,2): feat[f"{a}/{b}"]=raw[a]/raw[b]   # (i) ratios
for a,b in itertools.combinations(raw,2): feat[f"{a}*{b}"]=raw[a]*raw[b]   # (i) products
ok=lambda v: np.all(np.isfinite(v))
for a,b in itertools.permutations(raw,2):                                  # (ii) only if a/b>1 for all 576 inputs
    r=raw[a]/raw[b]
    if np.all(r>1):
        feat[f"ln({a}/{b})"]=np.log(r); feat[f"({a}/{b})/ln({a}/{b})"]=r/np.log(r)
names1=list(feat); F1=np.column_stack([feat[k] for k in names1])
# drop exact duplicates
_,ui=np.unique(np.round(F1,12),axis=1,return_index=True); ui=np.sort(ui); names1=[names1[i] for i in ui]; F1=F1[:,ui]
print("primary features:",len(names1), "(ii)-type:",[k for k in names1 if k.startswith("ln(") ])
cols=[F1]; names=list(names1)
for i,j in itertools.combinations(range(len(names1)),2):                  # (iii) sums and differences
    cols.append(np.c_[F1[:,i]+F1[:,j],F1[:,i]-F1[:,j]]); names+= [f"{names1[i]} + {names1[j]}",f"{names1[i]} - {names1[j]}"]
F=np.column_stack(cols); good=np.all(np.isfinite(F),0); F=F[:,good]; names=[n for n,g in zip(names,good) if g]
print("candidates:",F.shape[1])
def stump_all(F,y):
    """best single-threshold accuracy for every column (both orientations); returns acc, thr, sign"""
    n=len(y); o=np.argsort(F,0,kind="stable"); Fs=np.take_along_axis(F,o,0); ys=y[o]
    c1=np.vstack([np.zeros((1,F.shape[1])),np.cumsum(ys,0)])       # positives among first k
    k=np.arange(n+1)[:,None]; P=y.sum()
    acc_lo=(c1+( (n-k)-(P-c1) ))/n        # predict 1 for x<=thr (first k), 0 for rest
    acc_hi=(( k-c1)+(P-c1))/n             # predict 0 for first k, 1 for rest
    valid=np.vstack([np.ones((1,F.shape[1]),bool),Fs[1:]>Fs[:-1],np.ones((1,F.shape[1]),bool)])  # split only between distinct values
    acc_lo=np.where(valid,acc_lo,-1); acc_hi=np.where(valid,acc_hi,-1)
    best_lo=acc_lo.max(0); best_hi=acc_hi.max(0); sign=np.where(best_lo>=best_hi,1,-1)
    kk=np.where(sign==1,acc_lo.argmax(0),acc_hi.argmax(0))
    Fpad=np.vstack([Fs[:1]-1,Fs,Fs[-1:]+1])
    thr=0.5*(Fpad[kk,np.arange(F.shape[1])]+Fpad[kk+1,np.arange(F.shape[1])])
    return np.maximum(best_lo,best_hi),thr,sign
def predict(f,thr,sign): return (f<=thr).astype(int) if sign==1 else (f>thr).astype(int)
tau=d.rX.values/d.rB.values-d.nA.values*(d.nA.values-(d.rA.values/d.rB.values)/np.log(d.rA.values/d.rB.values))
for name,A,B in (("oxides -> halides",oxi,hal),("halides -> oxides",hal,oxi)):
    acc,thr,sg=stump_all(F[A],y[A]); best=acc.max(); ties=np.where(acc>=best-1e-12)[0]
    test=np.array([np.mean(predict(F[B,j],thr[j],sg[j])==y[B]) for j in ties])
    j=ties[0]
    print(f"=== {name}: train n={A.sum()}, best train acc {best:.3f}; #candidates tied at best {len(ties)}")
    print(f"   first best: '{names[j]}' {'<=' if sg[j]==1 else '>'} {thr[j]:.4f} -> transfer acc {test[0]:.3f}")
    print(f"   transfer acc over tied best: min {test.min():.3f} median {np.median(test):.3f} max {test.max():.3f}")
    for jj,tt in list(zip(ties,test))[:8]: print(f"      {names[jj]:45s} thr {thr[jj]:.3f} sign {sg[jj]:+d} transfer {tt:.3f}")
    # top-50 by train accuracy: distribution of transfer
    top=np.argsort(-acc)[:50]; tt=np.array([np.mean(predict(F[B,j],thr[j],sg[j])==y[B]) for j in top])
    print(f"   top-50 train accs {acc[top].min():.3f}-{acc[top].max():.3f}; their transfer acc: min {tt.min():.3f} median {np.median(tt):.3f} max {tt.max():.3f}")
    # tau stump trained on same family
    a2,t2,s2=stump_all(tau[A,None],y[A]); print(f"   tau stump: train acc {a2[0]:.3f}, thr {t2[0]:.3f}, transfer {np.mean(predict(tau[B],t2[0],s2[0])==y[B]):.3f}")
    # in-family 'honest' check: best candidate chosen on a random half of the training family, evaluated on the other half
    rng=np.random.default_rng(0); idx=np.where(A)[0]; gaps=[]
    for r in range(20):
        p=rng.permutation(idx); h1,h2=p[:len(p)//2],p[len(p)//2:]
        a_,t_,s_=stump_all(F[h1],y[h1]); jb=a_.argmax()
        gaps.append((a_[jb],np.mean(predict(F[h2,jb],t_[jb],s_[jb])==y[h2])))
    g=np.array(gaps); print(f"   selection optimism within family (20 random halves): train-half best {g[:,0].mean():.3f}, other half {g[:,1].mean():.3f}, gap {np.mean(g[:,0]-g[:,1]):.3f}")
# null experiment: shuffled labels -> how high does the best training accuracy go?
rng=np.random.default_rng(1); best_null=[]
for r in range(20):
    ys=rng.permutation(y[oxi]); a_,_,_=stump_all(F[oxi],ys); best_null.append(a_.max())
print(f"null (oxide labels shuffled, 20x): best train acc {np.mean(best_null):.3f} +/- {np.std(best_null):.3f}; majority rate {max(y[oxi].mean(),1-y[oxi].mean()):.3f}")
print(f"time {time.time()-t0:.1f}s")
