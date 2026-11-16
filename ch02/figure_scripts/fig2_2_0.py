"""Fig 2.2.0: two failures of existence/uniqueness (sqrt(x) non-uniqueness; 100x-101e^{-t} ill-conditioning)."""
import sys; sys.path.insert(0,'../ch02/figures')
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# numerical check of the "실행 결과"
f=lambda t,x: 100*x-101*np.exp(-t)
for x0 in [1.0,1.0+1e-6]:
    s=solve_ivp(f,(0,0.3),[x0],method='LSODA',rtol=1e-12,atol=1e-14)
    print("x0=%.7f  x(0.3) numeric=%.4e  analytic=%.4e"%(x0,s.y[0,-1],np.exp(-0.3)+(x0-1)*np.exp(30)))
print("exp(30)=%.3e"%np.exp(30))
fig,axs=plt.subplots(1,2,figsize=(13,5.2))
ax=axs[0]; t=np.linspace(0,4,300)
ax.plot(t,0*t,color=bs.NAVY,lw=2.6,ls='--',label=r'$x(t)=0$')
ax.plot(t,(t/2)**2,color=bs.RED,lw=2.6,label=r'$x(t)=(t/2)^2$')
for tc in [1.0,2.0]:  # other members of the family: stay at 0 until tc, then grow
    ax.plot(t,np.where(t<tc,0,((t-tc)/2)**2),color=bs.GRAY,lw=1.4,ls=':')
ax.plot([0],[0],'o',color='k',ms=9,zorder=5)
ax.annotate('same initial point $x(0)=0$',xy=(0.03,0.05),xytext=(0.9,-0.9),fontsize=16,
            arrowprops=dict(arrowstyle='->',lw=1.2),annotation_clip=False)
ax.text(4.05,((4-1)/2)**2,'branch at $t=1$',color=bs.GRAY,fontsize=15,va='center')
ax.text(4.05,((4-2)/2)**2,'branch at $t=2$',color=bs.GRAY,fontsize=15,va='center')
ax.set_xlabel('t'); ax.set_ylabel('x(t)'); ax.set_ylim(-1.2,4.3); ax.set_xlim(-0.2,5.2); ax.set_xticks([0,1,2,3,4])
ax.legend(loc='upper left',frameon=False)
ax=axs[1]; t=np.linspace(0,0.3,400)
ax.semilogy(t,np.exp(-t),color=bs.NAVY,lw=2.6,ls='--',label=r'$x(0)=1$ (exact $e^{-t}$)')
ax.semilogy(t,np.exp(-t)+1e-6*np.exp(100*t),color=bs.RED,lw=2.6,label=r'$x(0)=1+10^{-6}$')
ax.plot([0.3],[np.exp(-0.3)+1e-6*np.exp(30)],'o',color=bs.RED,ms=9)
ax.annotate(r'$1.07\times10^{7}$',xy=(0.3,1.07e7),xytext=(0.17,1.5e6),fontsize=16,
            arrowprops=dict(arrowstyle='->',lw=1.2))
ax.set_xlabel('t'); ax.set_ylabel('x(t)'); ax.set_ylim(0.3,1e8)
ax.legend(loc='upper left',frameon=False)
for a,l in zip(axs,'ab'): bs.panel_label(a,l)
fig.tight_layout(w_pad=3)
fig.savefig('../ch02/figures/Fig2.2.0_lipschitz.png')
