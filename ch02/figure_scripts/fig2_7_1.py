import sys; sys.path.insert(0,'../../ch02/figure_scripts'); import bookstyle as bs
import numpy as np, matplotlib.pyplot as plt, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
d=np.load('lab2_15_data.npz'); xf,uf,err,xc=d['xf'],d['uf'],d['err'],d['xc']
fig,axs=plt.subplots(1,2,figsize=(14,5.4))
ax=axs[0]
ax.plot(xf,np.sin(np.pi*xf),'-',color=bs.GRAY,lw=6,alpha=0.5,label=r"exact sin($\pi$x)")
ax.plot(xf,uf,'--',color=bs.NAVY,lw=2.4,label="PINN u$_\\theta$(x)")
ax.plot(xc,np.full_like(xc,-0.06),'|',color=bs.BLUE,ms=14,mew=1.6,label="collocation points (80)")
ax.set_ylim(-0.12,1.12); ax.set_xlabel("x"); ax.set_ylabel("u(x)")
ax.legend(frameon=False,loc='lower center',fontsize=14,bbox_to_anchor=(0.5,0.08))
bs.panel_label(ax,'a')
ax=axs[1]
ax.semilogy(xf,np.maximum(err,1e-16),'-',color=bs.RED,lw=2)
ax.set_xlabel("x"); ax.set_ylabel(r"|u$_\theta$ - u$_{exact}$|")
ax.text(0.03,0.93,f"max error = {err.max():.1e}",transform=ax.transAxes,fontsize=16)
ax.set_ylim(1e-13,1e-7)
bs.panel_label(ax,'b')
fig.tight_layout(); fig.savefig("Fig2.7.1_pinn.png")
