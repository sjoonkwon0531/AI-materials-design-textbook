import sys; sys.path.insert(0,'../../ch02/figure_scripts'); import bookstyle as bs
import numpy as np, matplotlib.pyplot as plt, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
d=np.load('lab2_14_data.npz'); t=d['t']; F=d['F']; M=d['M']; Lt=d['Lt']
fig,axs=plt.subplots(1,3,figsize=(19,5.4))
ax=axs[0]
ax.loglog(t[1:],F[1:],'-',color=bs.NAVY,lw=2.6,marker='o',markevery=[0,4,19,79,399],ms=7,label="F[c](t), sampled every 20 steps")
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
ax.yaxis.set_major_locator(FixedLocator([3,5,10,20,40])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("time t"); ax.set_ylabel("free energy F"); ax.legend(frameon=False,loc='upper right',fontsize=14)
ax.text(0.04,0.16,f"F(0) = {F[0]:.1f} (quench noise)",transform=ax.transAxes,fontsize=16)
ax.text(0.04,0.06,f"F increases: {(np.diff(F)>0).sum()} / {len(F)-1}",transform=ax.transAxes,fontsize=16)
bs.panel_label(ax,'a')
ax=axs[1]
ax.semilogy(t[1:],np.maximum(M[1:],1e-20),'o',color=bs.BLUE,ms=4,label="|mean(c) - mean(c$_0$)|")
ax.axhline(2.2e-16,color=bs.GRAY,ls='--',lw=1.8,label="machine epsilon")
ax.set_ylim(1e-20,1e-14); ax.text(0.04,0.06,f'max drift (every step) = 3.8e-17',transform=ax.transAxes,fontsize=15); ax.set_xlabel("time t"); ax.set_ylabel("mass drift")
ax.legend(frameon=False,loc='upper right',fontsize=14)
bs.panel_label(ax,'b')
ax=axs[2]
m=(t>=50)&(t<=400); p=np.polyfit(np.log(t[m]),np.log(Lt[m]),1)
ax.loglog(t[1:],Lt[1:],'o',color=bs.NAVY,ms=4,label="L(t) = 2π/<k>")
tt=np.array([50,400]); ax.loglog(tt,np.exp(p[1])*tt**p[0],'-',color=bs.RED,lw=2.6,label=f"fit 50 ≤ t ≤ 400: slope {p[0]:.2f}")
tr=np.array([5,400]); ax.loglog(tr,1.1*Lt[np.argmin(abs(t-5))]*(tr/5)**(1/3),'--',color=bs.GOLD,lw=2.4,label="t$^{1/3}$ (LSW)")
ax.set_xlabel("time t"); ax.set_ylabel("characteristic length L")
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
ax.yaxis.set_major_locator(FixedLocator([1,2,4,8])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(ScalarFormatter())
ax.legend(frameon=False,loc='upper left',fontsize=14)
bs.panel_label(ax,'c')
fig.tight_layout(); fig.savefig("Fig2.6.3_diagnostics.png")
print(p)
