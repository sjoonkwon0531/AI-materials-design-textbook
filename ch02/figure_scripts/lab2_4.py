"""Lab 2.4 / Fig 2.2.6: e=0.8 Kepler orbit with adaptive RK45 (GM=1, a=1, start at perihelion,
rtol=1e-8, atol=1e-11, 4 periods)."""
import sys; sys.path.insert(0,'../../ch02/figure_scripts')
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
e=0.8; P=2*np.pi; y0=[1-e,0,0,np.sqrt((1+e)/(1-e))]
def f(t,y):
    r3=(y[0]**2+y[1]**2)**1.5; return [y[2],y[3],-y[0]/r3,-y[1]/r3]
s=solve_ivp(f,(0,4*P),y0,method='RK45',rtol=1e-8,atol=1e-11)
dt=np.diff(s.t); r=np.hypot(s.y[0],s.y[1])
En=0.5*(s.y[2]**2+s.y[3]**2)-1/r
print("steps",len(dt),"nfev",s.nfev,"dt min %.4g max %.4g ratio %.0f"%(dt.min(),dt.max(),dt.max()/dt.min()))
# ratio excluding first & last step (first step chosen by initial guess, last truncated)
d=dt[1:-1]; print("interior ratio %.0f  min %.4g max %.4g"%(d.max()/d.min(),d.min(),d.max()))
print("interior (no start-up/final) min %.4g max %.4g ratio %.0f"%(dt[2:-1].min(),dt[2:-1].max(),dt[2:-1].max()/dt[2:-1].min()))
print("median dt near perihelion (r<0.25): %.4g ; near aphelion (r>1.75): %.4g"%(np.median(dt[r[:-1]<0.25]),np.median(dt[r[:-1]>1.75])))
print("energy E0=%.8f  E_end=%.8f rel drift=%.2e"%(En[0],En[-1],En[-1]/En[0]-1))
for per in [40]:
    s2=solve_ivp(f,(0,per*P),y0,method='RK45',rtol=1e-8,atol=1e-11)
    r2=np.hypot(s2.y[0],s2.y[1]); E2=0.5*(s2.y[2]**2+s2.y[3]**2)-1/r2
    print(per,"periods rel energy drift %.2e"%(E2[-1]/E2[0]-1))
fig,axs=plt.subplots(1,2,figsize=(15,5.4),gridspec_kw=dict(width_ratios=[1,1.3]))
ax=axs[0]
ax.plot(s.y[0],s.y[1],color=bs.GRAY,lw=1)
n1=np.searchsorted(s.t,P)
ax.plot(s.y[0][:n1+1],s.y[1][:n1+1],'o',ms=3.5,color=bs.NAVY,label='RK45 steps (1st period)')
ax.plot(0,0,'*',ms=18,color=bs.GOLD,mec='k',label='Central mass')
ax.annotate('perihelion',xy=(0.2,0),xytext=(-0.62,-0.2),fontsize=15,arrowprops=dict(arrowstyle='->'))
ax.annotate('aphelion',xy=(-1.8,0),xytext=(-1.55,0.12),fontsize=15,arrowprops=dict(arrowstyle='->'))
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_xlim(-2.0,0.6); ax.set_ylim(-0.75,0.95)
ax.legend(loc='upper left',frameon=False,fontsize=13,bbox_to_anchor=(-0.02,1.12))
ax=axs[1]
k=slice(2,-1)  # drop 2 start-up steps and the final truncated step
ax.semilogy(s.t[:-1][k]/P,dt[k],'-',color=bs.NAVY,lw=1.4,marker='o',ms=3.5,label=r'Step size $\Delta t$')
ax2=ax.twinx(); ax2.plot(s.t/P,r,color=bs.RED,ls='--',lw=1.8,label='Radius r'); ax2.set_ylabel('Radius r',color=bs.RED)
ax2.spines['right'].set_visible(True); ax2.spines['top'].set_visible(False); ax2.set_ylim(0,2.6)
ax.set_xlabel('Time (orbital periods)'); ax.set_ylabel(r'Step size $\Delta t$'); ax.set_ylim(1e-3,3); ax.set_xlim(-0.1,4.1)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,loc='upper center',frameon=False,ncol=2,fontsize=14)
bs.panel_label(axs[0],'a',x=-0.2,y=1.1); bs.panel_label(axs[1],'b',x=-0.12)
fig.tight_layout(w_pad=2)
fig.savefig('../../ch02/figures/Fig2.2.6_adaptive.png')
