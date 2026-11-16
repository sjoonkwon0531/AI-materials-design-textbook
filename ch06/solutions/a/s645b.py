from unmix_common import *
def metrics2(S,Se,X,F):
    cc,Sm,_=match(S,Se); Sm=Sm/np.abs(Sm).max(1,keepdims=True)   # same max-normalization convention as truth
    neg=np.maximum(np.array([-s[s<0].sum()/np.abs(s).sum() for s in Sm]),0)
    Fh=np.array([nnls(Sm.T,x)[0] for x in X]); Fh/=Fh.sum(1,keepdims=True)
    return cc,neg,np.mean(np.abs(Fh-F))
def run(label,P3x,fr="dir"):
    rng=np.random.default_rng(0)
    S=np.array([phase(P1),phase(P2),phase(P3x)]); F=rng.dirichlet([1,1,1],size=150)
    if fr=="mixed": F=0.2+0.4*F           # every sample contains >= 20% of every phase
    X=F@S+rng.normal(0,0.01,(150,600))
    print(f"--- {label}; min fraction in data = {F.min():.3f}; max fraction = {F.max():.3f}")
    cc,neg,fm=metrics2(S,ica(X),X,F); print("  ICA        : corr",fmt(cc)," neg",fmt(neg)," fracMAE %.3f"%fm)
    W,H,err=nmf(X); cc,neg,fm=metrics2(S,H,X,F); print("  NMF nndsvda: corr",fmt(cc)," neg",fmt(neg)," fracMAE %.3f  err %.4f"%(fm,err))
    Hs=[];fms=[];errs=[];ccs=[]
    for rs in range(5):
        W,H,err=nmf(X,init="random",rs=rs); cc,neg,fm=metrics2(S,H,X,F); _,Hm,_=match(S,H)
        Hs.append(Hm/np.linalg.norm(Hm,axis=1,keepdims=True)); fms.append(fm); errs.append(err); ccs.append(cc.min())
    dmax=max(np.linalg.norm(Hs[a]-Hs[b]) for a in range(5) for b in range(a+1,5))
    print("  NMF random 0-4: min corr per seed",fmt(ccs),"; fracMAE",fmt(fms),"; recon err",", ".join(f"{e:.4f}" for e in errs),"; max L2 diff between seeds %.4f"%dmax)
P3x=[(20.2,1.0),(31.2,0.4),(38.7,0.7),(47.2,0.5),(64.2,0.4)]
P3e=[(20.0,1.0),(31.0,0.4),(38.5,0.7),(47.0,0.5),(64.0,0.4)]
run("lab case A (reference)",P3)
run("phase-3 peaks 0.2 deg from phase 1/2 peaks",P3x)
run("phase-3 peaks 0.2 deg off, no near-pure samples",P3x,"mixed")
run("phase-3 peaks exactly on phase 1/2 peaks, no near-pure samples",P3e,"mixed")
run("phase-3 peaks exactly on phase 1/2 peaks, Dirichlet(1,1,1)",P3e)
