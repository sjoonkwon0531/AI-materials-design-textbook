import sys, importlib, numpy as np
sys.path.insert(0, "../../../ch08/labs")
E = importlib.import_module("81_anneal_env")
P, R = E.build_model(); nS, nA = E.nS, E.nA
np.set_printoptions(precision=6, suppress=True)
# 8.1.2
g=0.9; VA=1/(1-0.5*g); print("8.1.2 V(A)=",VA)
V=np.zeros(2)
for k in range(1,5):
    V=np.array([1+g*(0.5*V[0]+0.5*V[1]), g*V[1]]); print(k,V, abs(V[0]-VA))
Ppi=np.array([[0.5,0.5],[0,1]]); print("eig",np.linalg.eigvals(Ppi))
# 8.1.3 / 8.1.4
def solve(G):
    V=np.zeros(nS); k=0
    while True:
        Vn=(R+G*np.einsum("asp,p->sa",P,V)).max(1); k+=1
        d=np.abs(Vn-V).max(); V=Vn
        if d<1e-12: break
    pi=(R+G*np.einsum("asp,p->sa",P,V)).argmax(1)
    return V,pi,k
def peval(pi,G):
    Pp=P[pi,np.arange(nS)]; Rp=R[np.arange(nS),pi]
    return np.linalg.solve(np.eye(nS)-G*Pp,Rp), Pp, Rp
print("8.1.3")
V95,pi95,_=solve(0.95)
for G in [0.9,0.95,0.99,0.999]:
    _,Pp,Rp=peval(pi95,G); A=np.eye(nS)-G*Pp
    kap=np.linalg.cond(A,p=np.inf); ub=(1+G)/(1-G)
    # also with pi* at that gamma
    V,pi,_=solve(G); _,Pp2,_=peval(pi,G); kap2=np.linalg.cond(np.eye(nS)-G*Pp2,p=np.inf)
    nA_=np.abs(A).sum(1).max(); nAi=np.abs(np.linalg.inv(A)).sum(1).max()
    print(f"G={G} cond(pi*_0.95)={kap:.3f} cond(pi*_G)={kap2:.3f} bound={ub:.3f} |A|={nA_:.4f} |Ainv|={nAi:.3f} 1/(1-g)={1/(1-G):.1f} relerr bound={kap2*1e-3:.4g}")
    # actual worst-case relative error: perturb r by 1e-3 relative in sign of inverse
    Vp,Pp,Rp=peval(pi,G); Ai=np.linalg.inv(np.eye(nS)-G*Pp)
    i=np.argmax(np.abs(Ai).sum(1)) 
    dr=1e-3*np.abs(Rp).max()*np.sign(Ai[np.argmax(np.abs(Vp)) ]) 
    dV=Ai@dr; print("   ex. pert: |dV|/|V|=",np.abs(dV).max()/np.abs(Vp).max(), " |dr|/|r|",np.abs(dr).max()/np.abs(Rp).max())
print("8.1.4")
TEMPS=E.TEMPS.astype(int)
for G in [0.8,0.95,0.99]:
    V,pi,k=solve(G)
    V1100,_,_=peval(np.full(nS,3),G)
    print(f"G={G} iters={k} pred={np.log(1e-12)/np.log(G):.0f} T*={TEMPS[pi].tolist()}")
    print(f"   V*(20)={V[20]:.4f} V1100(20)={V1100[20]:.4f} diff={V[20]-V1100[20]:.4f} rel={(V[20]-V1100[20])/abs(V1100[20]):.4f}")
    print(f"   V*(5)={V[5]:.4f} V1100(5)={V1100[5]:.4f} diff={V[5]-V1100[5]:.4f} rel={(V[5]-V1100[5])/abs(V1100[5]):.4f}")
    print(f"   V*(0)={V[0]:.4f} V1100(0)={V1100[0]:.4f} diff={V[0]-V1100[0]:.4f}")
    # first delta
    d0=np.abs((R+G*np.einsum("asp,p->sa",P,np.zeros(nS))).max(1)).max()
    print("   ln(1e-12/d0)/ln G=",np.log(1e-12/d0)/np.log(G), " horizon 1/(1-g)=",1/(1-G))
# 8.1.5
from math import exp
s=0.5
# grad log pi: a=1: 1-s ; a=0: -s
p1=s; gl1=1-s; gl0=-s
bstar=(p1*gl1**2*1+(1-p1)*gl0**2*0)/(p1*gl1**2+(1-p1)*gl0**2)
def var(b):
    m=p1*gl1*(1-b)+(1-p1)*gl0*(0-b)
    m2=p1*(gl1*(1-b))**2+(1-p1)*(gl0*(0-b))**2
    return m, m2-m**2
print("8.1.5 b*=",bstar, "b=0:",var(0),"b*:",var(bstar),"b=0.5",var(0.5))
print("more digits")
for G in [0.8,0.95,0.99]:
    V,pi,k=solve(G); V1100,_,_=peval(np.full(nS,3),G)
    print(G, "diff20=%.3e diff5=%.4e"%(V[20]-V1100[20], V[5]-V1100[5]), "|V|inf=%.4f"%np.abs(V).max())
V,pi,k=solve(0.999); print("0.999 V(20)",V[20], "T*",TEMPS[pi].tolist())
