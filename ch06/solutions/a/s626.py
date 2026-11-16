import sys, importlib, numpy as np
sys.path.insert(0,'../ch06/labs'); importlib.import_module("62_dscribe_compat")
from dscribe.descriptors import ACSF
from ase import Atoms
r=1.09; RC=5.0
tet = r/np.sqrt(3)*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
sq = r*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
fc=lambda x: 0.5*(np.cos(np.pi*x/RC)+1)*(x<=RC)
def G4(nb,eta,zeta,lam,simple=False):
    s=0
    for j in range(4):
        for k in range(4):
            if j==k: continue
            rj,rk=nb[j],nb[k]; c=rj@rk/np.linalg.norm(rj)/np.linalg.norm(rk); rjk=np.linalg.norm(rj-rk)
            w=1.0 if simple else np.exp(-eta*(rj@rj+rk@rk+rjk**2))*fc(np.linalg.norm(rj))*fc(np.linalg.norm(rk))*fc(rjk)
            s+=(1+lam*c)**zeta*w
    return 2**(1-zeta)*s
for nm,nb in (("tet",tet),("sq",sq)):
    cs=[nb[j]@nb[k]/r/r for j in range(4) for k in range(4) if j!=k]
    print(nm,"sum cos=%.4f |sum rhat|^2-4=%.4f sum cos^2=%.4f"%(sum(cs),np.linalg.norm(nb.sum(0)/r)**2-4,sum(np.array(cs)**2)),
          "sum P2=%.4f"%(sum(1.5*np.array(cs)**2-0.5)+4))
    for z in (1,2):
        for lam in (1,-1):
            print(f"   simple zeta={z} lam={lam:+d}: {G4(nb,0,z,lam,True):.4f}   full(eta=0.005,rc=5): {G4(nb,0.005,z,lam):.4f}")
    rjk=sorted({round(np.linalg.norm(nb[j]-nb[k]),4) for j in range(4) for k in range(j+1,4)})
    print("   r_jk values", rjk, " exp(-eta r^2)fc(r) :", [round(np.exp(-0.005*d*d)*fc(d),4) for d in rjk])
# DScribe check: H-H pair channel of G4 for C center
g4=[[0.005,z,lam] for z in [1,2] for lam in [-1,1]]
a=ACSF(species=["C","H"],r_cut=RC,g4_params=g4)
for nm,nb in (("tet",tet),("sq",sq)):
    m=Atoms("CH4",positions=np.vstack([[0,0,0],nb]))
    v=a.create(m,centers=[0])[0]
    print("DScribe",nm,np.round(v,4))
