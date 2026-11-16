import sys; sys.path.insert(0,'../ch02/figures'); import bookstyle as bs
import numpy as np, matplotlib.pyplot as plt, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
def ftcs_step(u, lam):
    un=u.copy(); un[1:-1]=u[1:-1]+lam*(u[2:]-2*u[1:-1]+u[:-2]); return un
fig,axs=plt.subplots(1,3,figsize=(17,5.2))
h=0.25; x=np.arange(0,1+1e-9,h); u0=np.sin(np.pi*x); u0[0]=u0[-1]=0
xs=np.linspace(0,1,200)
# (a) stable lam=0.4
ax=axs[0]; ls=['-','--','-.',':','-']; mk=['o','s','^','D','v']
u=u0.copy(); k=0.4*h*h
for j,n in enumerate([0,2,4,8,16]):
    if j>0:
        for _ in range(n-[0,2,4,8,16][j-1]): u=ftcs_step(u,0.4)
    ax.plot(x,u,ls[j],marker=mk[j],ms=8,color=bs.CYCLE[j%6] if j<4 else bs.GREEN,lw=2,label=f"t = {n*k:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.set_ylim(-0.05,1.1)
ax.legend(frameon=False,loc='upper right',fontsize=14,title=r"$\lambda$ = 0.4, h = 0.25",title_fontsize=15)
bs.panel_label(ax,'a')
# (b) unstable lam=4
ax=axs[1]; u=u0.copy()
for n in range(4):
    ax.plot(x,u,ls[n],marker=mk[n],ms=8,lw=2,color=bs.CYCLE[n],label=f"step {n}")
    u=ftcs_step(u,4.0)
ax.axhline(0,color=bs.GRAY,lw=0.8)
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(frameon=False,loc='upper left',fontsize=14,ncol=2,title=r"$\lambda$ = 4, h = 0.25",title_fontsize=15)
ax.annotate("u(0.25) = -0.95\nafter one step",xy=(0.25,-0.9497),xytext=(0.6,-2.75),fontsize=15,
            arrowprops=dict(arrowstyle='->',color=bs.GRAY))
ax.set_ylim(-3.0,3.6)
bs.panel_label(ax,'b')
# (c) max|u| vs step, h=0.05 with 1e-4 sawtooth seed
ax=axs[2]; hf=0.05; xf=np.arange(0,1+1e-9,hf); N=len(xf)
v0=np.sin(np.pi*xf)+1e-4*(-1.0)**np.arange(N); v0[0]=v0[-1]=0
styles={0.4:('-','o',bs.NAVY),0.5:('--','s',bs.BLUE),0.75:('-.','^',bs.GOLD),4.0:(':','D',bs.RED)}
for lam,(l,m,c) in styles.items():
    v=v0.copy(); mx=[np.abs(v).max()]
    for n in range(40):
        v=ftcs_step(v,lam); mx.append(np.abs(v).max())
    ax.semilogy(range(41),mx,l,marker=m,markevery=4,ms=7,lw=2,color=c,label=rf"$\lambda$ = {lam:g}")
ax.set_ylim(1e-2,1e10); ax.set_xlabel("time step n"); ax.set_ylabel("max |u|")
ax.legend(frameon=False,loc='upper left',fontsize=14)
ax.text(0.97,0.04,"h = 0.05, 1e-4 sawtooth seed",transform=ax.transAxes,ha='right',fontsize=14)
bs.panel_label(ax,'c')
fig.tight_layout(); fig.savefig("Fig2.5.1_explicit_stability.png")
