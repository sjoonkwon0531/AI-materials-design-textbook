from unmix_common import *
P3o=[(25.5,1.0),(31.2,0.8),(41.0,0.7),(44.7,0.6),(60.0,0.4)]
for lab,p3 in (("separated",P3),("overlapping",P3o)):
    rng=np.random.default_rng(0); S=np.array([phase(P1),phase(P2),phase(p3)]); F=rng.dirichlet([1,1,1],size=150); X=F@S+rng.normal(0,0.01,(150,600))
    for nm,Se in (("ICA",ica(X)),("NMF",nmf(X)[1])):
        _,Sm,_=match(S,Se); raw=metrics(S,Se,X,F)[2]; Sn=Sm/np.abs(Sm).max(1,keepdims=True)
        Fh=np.array([nnls(Sn.T,x)[0] for x in X]); Fh/=Fh.sum(1,keepdims=True)
        print(lab,nm,"raw-scale fracMAE %.3f   max-normalized %.3f"%(raw,np.mean(np.abs(Fh-F))))
    print(lab,"max fraction %.3f, #samples with some f>0.9: %d"%(F.max(),(F.max(1)>0.9).sum()))
