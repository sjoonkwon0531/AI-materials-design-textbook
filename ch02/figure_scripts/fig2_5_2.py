import sys; sys.path.insert(0,'../../ch02/figure_scripts'); import bookstyle as bs
import numpy as np, matplotlib.pyplot as plt, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
th=np.linspace(0,np.pi,400); s2=np.sin(th/2)**2
fig,axs=plt.subplots(1,2,figsize=(14,5.4))
for ax in axs:
    ax.axhspan(-1,1,color=bs.GRAY,alpha=0.15,lw=0)
    ax.axvline(np.pi,color=bs.PURPLE,ls=':',lw=2)
    ax.set_xticks([0,np.pi/2,np.pi]); ax.set_xticklabels(["0",r"$\pi$/2",r"$\pi$"])
    ax.set_xlabel(r"mode phase $\theta = \xi h$"); ax.set_xlim(0,np.pi*1.02)
ax=axs[0]
for lam,l,c in ((0.25,'-',bs.NAVY),(0.5,'--',bs.BLUE),(0.7,'-.',bs.RED)):
    ax.plot(th,1-4*lam*s2,l,color=c,lw=2.6,label=rf"FTCS, $\lambda$ = {lam}")
ax.plot([np.pi],[1-4*0.7],'o',color=bs.RED,ms=9)
ax.annotate("g = -1.8 < -1\nunstable",xy=(np.pi,-1.8),xytext=(1.55,-1.75),fontsize=15,
            arrowprops=dict(arrowstyle='->',color=bs.GRAY))
ax.text(np.pi-0.05,1.2,"Nyquist",color=bs.PURPLE,ha='right',fontsize=15)
ax.set_ylim(-2.0,1.4); ax.set_ylabel(r"amplification factor g($\theta$)")
ax.legend(frameon=False,loc='lower left',fontsize=14)
bs.panel_label(ax,'a')
ax=axs[1]
for lam,l,m in ((0.5,'-',None),(4.0,'--',None)):
    ax.plot(th,1/(1+4*lam*s2),l,color=bs.NAVY,lw=2.6,label=rf"BTCS, $\lambda$ = {lam:g}")
    ax.plot(th,(1-2*lam*s2)/(1+2*lam*s2),l,color=bs.GOLD,lw=2.6,marker='o',markevery=40,ms=7,label=rf"CN, $\lambda$ = {lam:g}")
ax.set_ylim(-2.0,1.4); ax.set_ylabel(r"g($\theta$)")
ax.legend(frameon=False,loc='lower left',fontsize=14,ncol=2)
bs.panel_label(ax,'b')
fig.tight_layout(); fig.savefig("Fig2.5.2_von_neumann.png")
