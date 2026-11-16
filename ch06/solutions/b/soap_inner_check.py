"""Ex 6.7.5(b) check: within cells of equal composition, spread of inner-averaged vs outer-averaged SOAP."""
import numpy as np
src=open("soap_mod.py").read(); exec(src[:src.index('print("(c)')])
nAu=np.round(x*32).astype(int)
for name,F in (("inner",S_inn),("outer",S_out)):
    rel=[]
    for k in np.unique(nAu):
        m=nAu==k
        if m.sum()>=3 and 0<k<32:
            Fm=F[m]; rel.append(np.linalg.norm(Fm-Fm.mean(0),axis=1).mean()/np.linalg.norm(Fm.mean(0)))
    print(name,"mean relative within-composition spread %.2e"%np.mean(rel))
