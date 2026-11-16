from unmix_common import *
P3x=[(20.2,1.0),(31.2,0.4),(38.7,0.7),(47.2,0.5),(64.2,0.4)]
rng=np.random.default_rng(0)
S=np.array([phase(P1),phase(P2),phase(P3x)]); F=rng.dirichlet([1,1,1],size=150); X=F@S+rng.normal(0,0.01,(150,600))
C=np.corrcoef(S); print("true corr 12 13 23: %.3f %.3f %.3f"%(C[0,1],C[0,2],C[1,2]))
# purity: channels where phase 3 exceeds 0.1 and is > 5x the other two
r=S[2]/(S[0]+S[1]+1e-12); print("channels where S3>0.1 and S3 > 2x(S1+S2): %d ; >5x: %d"%(((S[2]>0.1)&(r>2)).sum(),((S[2]>0.1)&(r>5)).sum()))
for i in range(3):
    others=np.delete(S,i,0).sum(0); rr=S[i]/(others+1e-12); print(f"  phase {i+1}: max S_i/(sum others) over channels with S_i>0.1 = {rr[S[i]>0.1].max():.2f}")
cc,neg,fm,_=metrics(S,ica(X),X,F); print("ICA: corr",fmt(cc),"neg",fmt(neg),"fracMAE %.3f"%fm)
W,H,err=nmf(X); cc,neg,fm,_=metrics(S,H,X,F); print("NMF nndsvda: corr",fmt(cc),"neg",fmt(neg),"fracMAE %.3f err %.4f"%(fm,err))
Hs=[]
for rs in range(5):
    W,H,err=nmf(X,init="random",rs=rs); cc,neg,fm,_=metrics(S,H,X,F); _,Hm,_=match(S,H); Hs.append(Hm/np.linalg.norm(Hm,axis=1,keepdims=True))
    print(f"NMF random_state={rs}: corr {fmt(cc)}  fracMAE {fm:.3f}  recon err {err:.4f}")
d=[min(np.corrcoef(Hs[a][i],Hs[b][i])[0,1] for i in range(3)) for a in range(5) for b in range(a+1,5)]
print("min pairwise corr of matched components across seeds: %.4f (max diff spectrum L2 %.3f)"%(min(d),max(np.linalg.norm(Hs[a]-Hs[b]) for a in range(5) for b in range(a+1,5))))
# reference: separated case NMF random seeds for contrast
S0=np.array([phase(P1),phase(P2),phase(P3)]); rng=np.random.default_rng(0); F0=rng.dirichlet([1,1,1],size=150); X0=F0@S0+rng.normal(0,0.01,(150,600))
Hs=[]
for rs in range(5):
    W,H,err=nmf(X0,init="random",rs=rs); _,Hm,_=match(S0,H); Hs.append(Hm/np.linalg.norm(Hm,axis=1,keepdims=True))
print("separated case: max L2 diff across 5 seeds %.4f"%max(np.linalg.norm(Hs[a]-Hs[b]) for a in range(5) for b in range(a+1,5)))
