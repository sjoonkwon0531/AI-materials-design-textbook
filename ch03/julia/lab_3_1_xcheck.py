# numpy re-implementation of lab_3_1.jl logic (cross-check of printed numbers)
import numpy as np, itertools
AMU_G=1.66053906660e-24; M={"Si":28.0855,"Fe":55.845,"O":15.9994}
a=5.43; lat=a*np.eye(3)
fcc=[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]]; basis=[[0,0,0],[.25,.25,.25]]
frac=[np.mod(np.add(t,b),1.0) for t in fcc for b in basis]
key=lambda v: tuple(np.round(np.mod(np.round(v,8),1.0),6))
S={key(f) for f in frac}
vol=abs(np.linalg.det(lat)); dens=len(frac)*M["Si"]*AMU_G/(vol*1e-24)
Ws=[]
for p in itertools.permutations(range(3)):
    for s in itertools.product((-1,1),repeat=3):
        W=np.zeros((3,3)); 
        for r in range(3): W[r,p[r]]=s[r]
        Ws.append(W)
n=0
for W in Ws:
    for t in itertools.product(range(4),repeat=3):
        t=np.array(t)/4
        if all(key(W@f+t) in S for f in frac): n+=1
print("n_sites", len(frac)); print(f"density {dens:.3f}"); print("ops", n); print(f"vol {vol:.1f}")
print(f"Fe2O3 weight {2*M['Fe']+3*M['O']:.2f}  xFe {2/5:.3f}")
