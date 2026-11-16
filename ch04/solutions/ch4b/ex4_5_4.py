# Ex 4.5.4: train only on r in [1.2,2.5] -> extrapolation failure at r<1.2, consequences for MD
import numpy as np
def U_true(r): return 4.0*(r**-12 - r**-6)
def F_true(r): return 48.0*r**-13 - 24.0*r**-7
class KRR_U:
    def __init__(s, gamma=2.0, lam=1e-8): s.gamma, s.lam = gamma, lam
    def fit(s, r, y):
        s.r=r.copy(); K=np.exp(-s.gamma*(r[:,None]-r[None,:])**2)
        s.alpha=np.linalg.solve(K+s.lam*np.eye(len(r)), y); return s
    def U(s,q): return np.exp(-s.gamma*(q[:,None]-s.r[None,:])**2)@s.alpha
    def F(s,q):
        d=q[:,None]-s.r[None,:]; return -((np.exp(-s.gamma*d**2)*(-2*s.gamma*d))@s.alpha)
np.random.seed(2)
r_full = np.random.uniform(0.92,2.5,160); r_nar = np.random.uniform(1.2,2.5,160)
m_full = KRR_U().fit(r_full,U_true(r_full)); m_nar = KRR_U().fit(r_nar,U_true(r_nar))
print("  r     F_true     F_full(0.92-2.5)  F_narrow(1.2-2.5)   U_true   U_narrow")
for r in [0.80,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.30]:
    q=np.array([r]); print(f" {r:.2f} {F_true(q)[0]:9.2f}  {m_full.F(q)[0]:12.2f}  {m_nar.F(q)[0]:14.2f}  {U_true(q)[0]:9.3f} {m_nar.U(q)[0]:8.3f}")
rr=np.linspace(0.85,1.2,701); Fn=m_nar.F(rr)
print(f"narrow model: max learned F on [0.85,1.2] = {Fn.max():.2f}; true F at 0.85 = {F_true(np.array([0.85]))[0]:.1f}")
# --- MD (Lab 4.4/4.5 engine; 2D, N=64, rho=0.7, T=0.7, rescale every 20 steps) ---
rc2=2.5**2
def init(nside=8,density=0.7):
    N=nside*nside; L=np.sqrt(N/density); a=L/nside; xs=(np.arange(nside)+0.5)*a
    X,Y=np.meshgrid(xs,xs); return np.stack([X.ravel(),Y.ravel()],1),L,N
def forces(pos,L,Ffun,rmin2=0.0):
    N=len(pos); F=np.zeros_like(pos)
    for i in range(N-1):
        rij=pos[i]-pos[i+1:]; rij-=L*np.round(rij/L); r2=(rij*rij).sum(1)
        m=(r2<rc2)&(r2>rmin2); rr=np.sqrt(r2[m]); fv=rij[m]*(Ffun(rr)/rr)[:,None]
        F[i]+=fv.sum(0); np.add.at(F,np.where(m)[0]+i+1,-fv)
    return F
def pairdist(pos,L):
    d=[]; 
    for i in range(len(pos)-1):
        rij=pos[i]-pos[i+1:]; rij-=L*np.round(rij/L); d.append(np.sqrt((rij*rij).sum(1)))
    return np.concatenate(d)
def run(Ffun,nsteps=4000,dt=0.004,T0=0.7,seed=0,rmin2=0.0):
    rng=np.random.default_rng(seed); pos,L,N=init(); vel=rng.standard_normal((N,2)); vel-=vel.mean(0)
    vel*=np.sqrt(T0/((vel*vel).sum()/(2*N-2))); F=forces(pos,L,Ffun,rmin2)
    dmin=[]; frac=[]; Utrue=[]
    for s in range(nsteps):
        vel+=0.5*dt*F; pos=(pos+dt*vel)%L; F=forces(pos,L,Ffun,rmin2); vel+=0.5*dt*F
        if not np.isfinite(pos).all(): return s, None
        T=(vel*vel).sum()/(2*N-2)
        if s%20==0: vel*=np.sqrt(T0/T)
        if s>=1500 and s%50==0:
            d=pairdist(pos,L); dmin.append(d.min()); frac.append(np.mean(d<1.0)*len(d)/N)
            dd=d[d<2.5]; Utrue.append(np.sum(U_true(np.maximum(dd,0.3)))/N)
    return nsteps, (np.min(dmin), np.mean(dmin), np.mean(frac), np.mean(Utrue))
for name,Ff in [("true LJ",F_true),("ML full range",m_full.F),("ML narrow [1.2,2.5]",m_nar.F)]:
    n,res=run(Ff)
    if res is None: print(f"{name:22s}: blew up at step {n}"); continue
    print(f"{name:22s}: min r_ij (overall) = {res[0]:.3f}, <min r_ij> = {res[1]:.3f}, "
          f"pairs with r<1.0 per atom = {res[2]:.3f}, true-LJ PE/atom along trajectory = {res[3]:.3f}")
