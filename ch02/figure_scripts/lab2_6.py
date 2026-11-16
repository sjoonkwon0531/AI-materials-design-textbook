"""Lab 2.6 / Fig 2.3: Cartesian pendulum (m=L=1, g=9.81), index-1 reduction naive vs Baumgarte.
Horizontal release: (x,y,vx,vy)=(1,0,0,0). RK45, rtol=1e-7 (atol default 1e-6 unless noted), alpha=beta=5."""
import sys; sys.path.insert(0,'../ch02/figures')
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
g=9.81
def make(alpha,beta):
    def f(t,u):
        x,y,vx,vy=u; r2=x*x+y*y; v2=vx*vx+vy*vy; phi=r2-1; dphi=2*(x*vx+y*vy)
        lam=(v2-g*y+alpha*dphi+0.5*beta**2*phi)/r2
        return [vx,vy,-lam*x,-lam*y-g]
    return f
u0=[1.0,0.0,0.0,0.0]; T=20.0; te=np.linspace(0,T,4001)
def run(a,b,atol=1e-6):
    s=solve_ivp(make(a,b),(0,T),u0,method='RK45',rtol=1e-7,atol=atol,t_eval=te)
    return s, np.abs(s.y[0]**2+s.y[1]**2-1)
for atol in [1e-6,1e-9]:
    sn,vn=run(0,0,atol)
    for ab in [5,10]:
        sb,vb=run(ab,ab,atol)
        print("atol=%g a=b=%d: naive |phi|(20)=%.2e max=%.2e nfev=%d | Baum |phi|(20)=%.2e max=%.2e nfev=%d | ratio@20=%.1e ratio(max)=%.1e"%(
            atol,ab,vn[-1],vn.max(),sn.nfev,vb[-1],vb.max(),sb.nfev,vn[-1]/vb[-1],vn.max()/vb.max()))
# monotonic? check running max of naive
sn,vn=run(0,0); sb,vb=run(5,5)
w=vn[::200]; print("naive sampled every 1 s:",["%.1e"%q for q in w])
print("naive 1s-window max:",["%.1e"%vn[i:i+200].max() for i in range(0,4000,400)])
print("Baum 1s-window max:",["%.1e"%vb[i:i+200].max() for i in range(0,4000,400)])
fig,ax=plt.subplots(figsize=(9.5,5.4))
ax.semilogy(te,np.maximum(vn,1e-16),color=bs.RED,lw=1.8,label='Naive index reduction')
ax.semilogy(te,np.maximum(vb,1e-16),color=bs.NAVY,lw=1.8,ls='--',label=r'Baumgarte ($\alpha=\beta=5$)')
ax.set_xlabel('t'); ax.set_ylabel(r'Constraint violation $|x^2+y^2-1|$')
ax.set_ylim(1e-10,1e-1)
ax.legend(loc='upper left',frameon=False)
fig.tight_layout(); fig.savefig('../ch02/figures/Fig2.3_dae_drift.png')
