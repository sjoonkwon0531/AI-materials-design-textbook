import numpy as np
np.random.seed(1)
# Lab 4.4  MD with Velocity Verlet (recovered from v1.0 session).
def sho_run(method, dt, nsteps, x0=1.0, v0=0.0):
    x, v = x0, v0; X=[x]; V=[v]; E=[0.5*(v*v+x*x)]
    for _ in range(nsteps):
        if method=="verlet":
            v += 0.5*dt*(-x); x += dt*v; v += 0.5*dt*(-x)
        elif method=="euler":
            xn = x + dt*v; vn = v + dt*(-x); x,v = xn,vn
        elif method=="symplectic_euler":
            v += dt*(-x); x += dt*v
        X.append(x); V.append(v); E.append(0.5*(v*v+x*x))
    return np.array(X), np.array(V), np.array(E)

dt_sho=0.15; n_sho=300
Xv,Vv,Ev = sho_run("verlet", dt_sho, n_sho)
Xe,Ve,Ee = sho_run("euler",  dt_sho, n_sho)
Xs,Vs,Es = sho_run("symplectic_euler", dt_sho, n_sho)
print("=== Part A: harmonic oscillator, energy after",n_sho,"steps (exact=0.5) ===")
print(f"  Velocity Verlet : E_end={Ev[-1]:.4f}   drift={(Ev[-1]-Ev[0])/Ev[0]:+.2e}")
print(f"  Symplectic Euler: E_end={Es[-1]:.4f}   drift={(Es[-1]-Es[0])/Es[0]:+.2e}")
print(f"  Forward Euler   : E_end={Ee[-1]:.4f}   drift={(Ee[-1]-Ee[0])/Ee[0]:+.2e}  (blows up)")

rc=2.5; rc2=rc*rc; ushift=4.0*(rc**-12 - rc**-6)
def init_lattice(nside, density):
    N=nside*nside; L=np.sqrt(N/density); a=L/nside
    xs=(np.arange(nside)+0.5)*a; X,Y=np.meshgrid(xs,xs)
    return np.stack([X.ravel(),Y.ravel()],axis=1), L, N
def forces_energy(pos,L):
    N=len(pos); F=np.zeros_like(pos); PE=0.0
    for i in range(N-1):
        rij=pos[i]-pos[i+1:]; rij-=L*np.round(rij/L)
        r2=np.sum(rij*rij,axis=1); m=r2<rc2
        r2m=r2[m]; rijm=rij[m]; inv2=1.0/r2m; inv6=inv2**3
        fs=(48.0*inv6*inv6-24.0*inv6)*inv2
        fv=rijm*fs[:,None]; F[i]+=fv.sum(axis=0)
        np.add.at(F, np.where(m)[0]+(i+1), -fv)
        PE+=np.sum(4.0*(inv6*inv6-inv6)-ushift)
    return F,PE
def kinetic_temp(vel):
    KE=0.5*np.sum(vel*vel); N=len(vel); return KE, 2.0*KE/(2*N-2)
def run_md(nsteps,dt,nside=6,density=0.7,T0=1.0,thermostat=None):
    pos,L,N=init_lattice(nside,density); vel=np.random.randn(N,2); vel-=vel.mean(axis=0)
    _,T=kinetic_temp(vel); vel*=np.sqrt(T0/T); F,PE=forces_energy(pos,L)
    h={"t":[],"KE":[],"PE":[],"E":[],"T":[]}
    for s in range(nsteps):
        vel+=0.5*dt*F; pos=(pos+dt*vel)%L; F,PE=forces_energy(pos,L); vel+=0.5*dt*F
        KE,T=kinetic_temp(vel)
        if thermostat=="rescale" and s%20==0 and T>0: vel*=np.sqrt(T0/T); KE,T=kinetic_temp(vel)
        if s%5==0:
            h["t"].append(s*dt);h["KE"].append(KE);h["PE"].append(PE);h["E"].append(KE+PE);h["T"].append(T)
    return {k:np.array(v) for k,v in h.items()}

dt=0.004; nsteps=6000
print("\n=== Part B: 2D LJ NVE (Velocity Verlet), N=36, rho=0.7 ===")
nve=run_md(nsteps,dt)
pp=(nve["E"].max()-nve["E"].min())/abs(nve["E"].mean())
print(f"  total E mean={nve['E'].mean():.3f}, peak-to-peak/|E|={pp:.2e}, E(end)/E(0)={nve['E'][-1]/nve['E'][0]:.4f}")
print("=== Part C: NVT velocity-rescaling thermostat, T0=1.0 ===")
nvt=run_md(nsteps,dt,thermostat="rescale",T0=1.0); half=len(nvt["T"])//2
print(f"  NVT mean T (2nd half)={nvt['T'][half:].mean():.3f} +/- {nvt['T'][half:].std():.3f}  (target 1.000)")
print(f"  NVE mean T (2nd half)={nve['T'][half:].mean():.3f}  (uncontrolled)")
