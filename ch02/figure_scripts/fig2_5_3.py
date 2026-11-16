import sys; sys.path.insert(0,'../../ch02/figure_scripts'); import bookstyle as bs
import numpy as np, matplotlib.pyplot as plt, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
fig,axs=plt.subplots(1,2,figsize=(14,5.6))
ax=axs[0]
# grid: x_j columns, time levels n, n+1 ; upwind stencil sees x_{j-1}, x_j
for xj in range(-3,2):
    ax.plot([xj,xj],[0,1],color=bs.ICE,lw=1)
    ax.plot(xj,0,'o',color=bs.GRAY,ms=8)
ax.plot(0,1,'o',color=bs.NAVY,ms=13)
ax.text(0.08,1.06,r"$u_j^{n+1}$",fontsize=18)
ax.fill_between([-1,0],[0,0],[0,0],color=bs.GREEN)
# numerical domain of dependence: triangle to x_{j-1}..x_j
ax.plot([0,-1],[1,0],'-',color=bs.GREEN,lw=2.2); ax.plot([0,0],[1,0],'-',color=bs.GREEN,lw=2.2)
ax.plot([-1,0],[0,0],'-',color=bs.GREEN,lw=7,alpha=0.6,label="numerical stencil (upwind)")
# characteristics
ax.plot([0,-0.8],[1,0],'-',color=bs.BLUE,lw=3,label=r"characteristic, $\nu$ = 0.8 (inside)")
ax.plot([0,-1.3],[1,0],'--',color=bs.RED,lw=3,label=r"characteristic, $\nu$ = 1.3 (outside)")
ax.plot(-0.8,0,'^',color=bs.BLUE,ms=11); ax.plot(-1.3,0,'v',color=bs.RED,ms=11)
ax.set_xticks([-3,-2,-1,0,1]); ax.set_xticklabels([r"$x_{j-3}$",r"$x_{j-2}$",r"$x_{j-1}$",r"$x_j$",r"$x_{j+1}$"])
ax.set_yticks([0,1]); ax.set_yticklabels([r"$t^n$",r"$t^{n+1}$"])
ax.set_ylim(-0.1,1.75); ax.set_xlim(-3.3,1.3)
ax.legend(frameon=False,loc='upper left',fontsize=14)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
bs.panel_label(ax,'a',x=-0.06)
ax=axs[1]
dx=np.logspace(-3,-1,50)
ax.loglog(dx,dx**2/2,'-',color=bs.NAVY,lw=2.8,label=r"parabolic: $\Delta t \leq \Delta x^2/2\alpha$")
ax.loglog(dx,dx/1.0,'--',color=bs.RED,lw=2.8,label=r"hyperbolic: $\Delta t \leq \Delta x/|a|$")
ax.text(3e-3,1.2e-6*2.2,"slope 2",fontsize=16,color=bs.NAVY,rotation=0)
ax.text(3e-3,3e-3*2.2,"slope 1",fontsize=16,color=bs.RED)
ax.set_xlabel(r"grid spacing $\Delta x$"); ax.set_ylabel(r"max stable $\Delta t$")
ax.text(0.97,0.04,r"$\alpha$ = 1, |a| = 1",transform=ax.transAxes,ha='right',fontsize=15)
ax.legend(frameon=False,loc='upper left',fontsize=14)
bs.panel_label(ax,'b')
fig.tight_layout(); fig.savefig("Fig2.5.3_cfl.png")
