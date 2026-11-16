# numpy mirror of lab_2_4.jl using the rk45 re-implementation (validated against solve_ivp)
import numpy as np
exec(open('rk45_scipy_xcheck.py').read().split('# Lab 2.4')[0])
kep=lambda t,Y: np.array([Y[2],Y[3],-Y[0]/(Y[0]**2+Y[1]**2)**1.5,-Y[1]/(Y[0]**2+Y[1]**2)**1.5])
P=2*np.pi; y0=np.array([0.2,0,0,np.sqrt(1.8/0.2)])
ts,ys,nf,rej=rk45(kep,0,y0,4*P,1e-8,1e-11); dt=np.diff(ts)
print(len(dt),nf,rej)
for k in range(4):
    m=(ts[:-1]>=k*P)&(ts[:-1]<(k+1)*P); d=dt[m]; print(k+1,"%.4f %.4f %.0f"%(d.min(),d.max(),d.max()/d.min()))
r=np.hypot(ys[:,0],ys[:,1]); print("%.4f %.4f"%(np.median(dt[r[:-1]<0.25]),np.median(dt[r[:-1]>1.75])))
print("%.2e"%np.linalg.norm(ys[-1,:2]-y0[:2]))
En=0.5*(ys[:,2]**2+ys[:,3]**2)-1/r; print("%.8f %.8f %.2e"%(En[0],En[-1],En[-1]/En[0]-1))
