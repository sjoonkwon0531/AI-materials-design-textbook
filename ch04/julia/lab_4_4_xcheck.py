# Part A of lab_4_4.jl is deterministic: mirror it (note forward Euler updates x,v simultaneously)
def run(m,dt=0.15,n=300):
    x,v=1.0,0.0; E=[0.5*(v*v+x*x)]
    for _ in range(n):
        if m=="verlet": v+=0.5*dt*(-x); x+=dt*v; v+=0.5*dt*(-x)
        elif m=="euler": x,v=x+dt*v, v+dt*(-x)
        else: v+=dt*(-x); x+=dt*v
        E.append(0.5*(v*v+x*x))
    return E
for m in ["verlet","symplectic_euler","euler"]:
    E=run(m); print(f"{m:18s} E_end={E[-1]:.4f} drift={(E[-1]-E[0])/E[0]:+.2e}")
# Part B/C: ported LJ engine with the 2xN layout, run with numpy RNG seed 1 -> must equal chapter (same RNG)
import numpy as np
np.random.seed(1)  # consume the same RNG stream as lab_4_4.py (Part A uses none)
rc=2.5; rc2=rc*rc; ush=4*(rc**-12-rc**-6)
def init(ns,d):
    N=ns*ns; L=np.sqrt(N/d); a=L/ns; xs=(np.arange(ns)+0.5)*a; pos=np.zeros((2,N)); k=0
    for i in range(ns):
        for j in range(ns): pos[0,k]=xs[j]; pos[1,k]=xs[i]; k+=1
    return pos,L,N
def fe(pos,L):
    N=pos.shape[1]; F=np.zeros((2,N)); PE=0.0
    for i in range(N-1):
        d=pos[:,i:i+1]-pos[:,i+1:]; d-=L*np.round(d/L); r2=(d*d).sum(0); m=r2<rc2
        inv2=1/r2[m]; inv6=inv2**3; fs=(48*inv6*inv6-24*inv6)*inv2; fv=d[:,m]*fs
        F[:,i]+=fv.sum(1); idx=np.where(m)[0]+i+1; F[:,idx]-=fv; PE+=np.sum(4*(inv6*inv6-inv6)-ush)
    return F,PE
def kt(v): KE=0.5*np.sum(v*v); N=v.shape[1]; return KE,2*KE/(2*N-2)
def md(nsteps,dt,T0=1.0,th=None):
    pos,L,N=init(6,0.7); vel=np.random.randn(N,2).T.copy(); vel-=vel.mean(1,keepdims=True)
    _,T=kt(vel); vel*=np.sqrt(T0/T); F,PE=fe(pos,L); H={"E":[],"T":[]}
    for s in range(nsteps):
        vel+=0.5*dt*F; pos=(pos+dt*vel)%L; F,PE=fe(pos,L); vel+=0.5*dt*F; KE,T=kt(vel)
        if th and s%20==0 and T>0: vel*=np.sqrt(T0/T); KE,T=kt(vel)
        if s%5==0: H["E"].append(KE+PE); H["T"].append(T)
    return {k:np.array(v) for k,v in H.items()}
nve=md(6000,0.004); E=nve["E"]; print(f"NVE mean={E.mean():.3f} pp={(E.max()-E.min())/abs(E.mean()):.2e} ratio={E[-1]/E[0]:.4f}")
nvt=md(6000,0.004,th=1); h=len(nvt["T"])//2; print(f"NVT {nvt['T'][h:].mean():.3f} +/- {nvt['T'][h:].std():.3f}; NVE T {nve['T'][h:].mean():.3f}")
