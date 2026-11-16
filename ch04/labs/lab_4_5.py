import numpy as np
np.random.seed(2)
# Lab 4.5  DFT->ML bridge (minimal), recovered from v1.0 session (with g(r) normalization fix).
def U_true(r):  return 4.0*(r**-12 - r**-6)
def F_true(r):  return 48.0*r**-13 - 24.0*r**-7
def make_data(n):
    r = np.random.uniform(0.92, 2.5, n)
    return r, U_true(r)
class KRR_U:
    def __init__(self, gamma=2.0, lam=1e-8):
        self.gamma=gamma; self.lam=lam
    def fit(self, r, y):
        self.r=r.copy()
        K = np.exp(-self.gamma*(r[:,None]-r[None,:])**2)
        self.alpha = np.linalg.solve(K + self.lam*np.eye(len(r)), y)
        return self
    def U(self, rq):
        K = np.exp(-self.gamma*(rq[:,None]-self.r[None,:])**2)
        return K @ self.alpha
    def F(self, rq):
        d = rq[:,None]-self.r[None,:]
        K = np.exp(-self.gamma*d**2)
        dUdr = (K*(-2*self.gamma*d)) @ self.alpha
        return -dUdr

rg = np.linspace(0.95, 2.5, 400)
print("--- learning curve: U/force RMSE vs #training points ---")
print("  Ntrain   U-RMSE     F-RMSE")
learn=[]
for n in [10, 20, 40, 80, 160, 320]:
    rt, ut = make_data(n)
    m = KRR_U().fit(rt, ut)
    uerr = np.sqrt(np.mean((m.U(rg)-U_true(rg))**2))
    ferr = np.sqrt(np.mean((m.F(rg)-F_true(rg))**2))
    learn.append((n,uerr,ferr))
    print(f"  {n:5d}   {uerr:.3e}  {ferr:.3e}")
rt, ut = make_data(160); model = KRR_U().fit(rt, ut)
print(f"\nfinal model (160 pts): U-RMSE={np.sqrt(np.mean((model.U(rg)-U_true(rg))**2)):.2e}, "
      f"F-RMSE={np.sqrt(np.mean((model.F(rg)-F_true(rg))**2)):.2e}")

rc=2.5; rc2=rc*rc
def init_lattice(nside,density):
    N=nside*nside; L=np.sqrt(N/density); a=L/nside
    xs=(np.arange(nside)+0.5)*a; X,Y=np.meshgrid(xs,xs)
    return np.stack([X.ravel(),Y.ravel()],axis=1),L,N
def md_forces(pos,L,kind):
    N=len(pos); F=np.zeros_like(pos)
    for i in range(N-1):
        rij=pos[i]-pos[i+1:]; rij-=L*np.round(rij/L)
        r2=np.sum(rij*rij,axis=1); m=(r2<rc2)&(r2>0.81)
        rr=np.sqrt(r2[m]); rijm=rij[m]
        fr=F_true(rr) if kind=="true" else model.F(rr)
        fv=rijm*(fr/rr)[:,None]; F[i]+=fv.sum(axis=0)
        np.add.at(F, np.where(m)[0]+(i+1), -fv)
    return F
def run_md(kind,nsteps=4000,dt=0.004,nside=8,density=0.7,T0=0.7,collect_after=1500):
    pos,L,N=init_lattice(nside,density); vel=np.random.randn(N,2); vel-=vel.mean(0)
    KE=0.5*np.sum(vel*vel); T=2*KE/(2*N-2); vel*=np.sqrt(T0/T)
    F=md_forces(pos,L,kind)
    nbins=100; rmax=L/2; gr=np.zeros(nbins); nsnap=0
    edges=np.linspace(0,rmax,nbins+1)
    for s in range(nsteps):
        vel+=0.5*dt*F; pos=(pos+dt*vel)%L; F=md_forces(pos,L,kind); vel+=0.5*dt*F
        KE=0.5*np.sum(vel*vel); T=2*KE/(2*N-2)
        if s%20==0: vel*=np.sqrt(T0/T)
        if s>=collect_after and s%10==0:
            dd=[]
            for i in range(N-1):
                rij=pos[i]-pos[i+1:]; rij-=L*np.round(rij/L)
                dd.append(np.sqrt(np.sum(rij*rij,axis=1)))
            dall=np.concatenate(dd); dall=dall[dall<rmax]
            h,_=np.histogram(dall,bins=nbins,range=(0,rmax)); gr+=h; nsnap+=1
    rc_mid=0.5*(edges[1:]+edges[:-1])
    ring=np.pi*(edges[1:]**2-edges[:-1]**2); rho=N/L**2
    g=(gr/nsnap)/(((N-1)/2)*rho*ring)
    return rc_mid,g
r_t,g_t=run_md("true"); r_m,g_m=run_md("ml")
ip=np.argmax(g_t); ipm=np.argmax(g_m)
print(f"  true-LJ g(r) first peak: r={r_t[ip]:.3f}, g={g_t[ip]:.3f}")
print(f"  ML      g(r) first peak: r={r_m[ipm]:.3f}, g={g_m[ipm]:.3f}")
print(f"  RMSE between g(r) curves (r>0.9): {np.sqrt(np.nanmean((g_t[r_t>0.9]-g_m[r_t>0.9])**2)):.3f}")
print(f"  g(r) tail mean (r>3): true={np.nanmean(g_t[r_t>3]):.3f}, ml={np.nanmean(g_m[r_m>3]):.3f}")
