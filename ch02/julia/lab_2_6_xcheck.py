import numpy as np
exec(open('rk45_scipy_xcheck.py').read().split('# Lab 2.4')[0])
g=9.81
def mk(a,b):
    def f(t,u):
        x,y,vx,vy=u;r2=x*x+y*y;v2=vx*vx+vy*vy;phi=r2-1;dphi=2*(x*vx+y*vy);lam=(v2-g*y+a*dphi+0.5*b**2*phi)/r2
        return np.array([vx,vy,-lam*x,-lam*y-g])
    return f
R={}
for n,ab in [("naive",0.),("Baumgarte",5.)]:
    ts,us,nf,_=rk45(mk(ab,ab),0,[1.,0,0,0],20.,1e-7,1e-6);v=np.abs(us[:,0]**2+us[:,1]**2-1);R[n]=(ts,v)
    print(n,"%.2e %.2e %d"%(v[-1],v.max(),nf))
print("%.1e"%(R['naive'][1][-1]/R['Baumgarte'][1][-1]))
for n in R:
    ts,v=R[n];print(n," ".join("%.1e"%v[(ts>=t0)&(ts<t0+1)].max() for t0 in range(0,20,2)))
