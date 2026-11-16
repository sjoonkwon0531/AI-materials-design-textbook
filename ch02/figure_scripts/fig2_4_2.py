"""Fig 2.4.2: P1 FEM for -u''=pi^2 sin(pi x). L2 error by fine-grid interpolation (20 sub-points/element)."""
import sys; sys.path.insert(0,'../ch02/figures')
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
exec(open('../ch02/labs/b03_L1217.py').read())
from scipy.sparse import diags; from scipy.sparse.linalg import spsolve
def l2_err(x,u,nf=20):
    xf=np.linspace(0,1,(len(x)-1)*nf+1); uf=np.interp(xf,x,u)
    return np.sqrt(np.trapezoid((uf-exact(xf))**2,xf))
def fdm(N):
    h=1/N; x=np.linspace(0,1,N+1); n=N-1
    A=diags([np.full(n-1,-1.),np.full(n,2.),np.full(n-1,-1.)],[-1,0,1],format='csr')/h**2
    u=np.zeros(N+1); u[1:N]=spsolve(A,ffun(x[1:N])); return x,u
Ns=np.array([4,8,16,32,64,128,256]); hs=1/Ns
fe=[];fn=[];de=[]
for N in Ns:
    x,u=fem_p1(N); fe.append(l2_err(x,u)); fn.append(np.abs(u-exact(x)).max())
    x,u=fdm(N); de.append(l2_err(x,u))
fe,fn,de=map(np.array,(fe,fn,de))
print("FEM L2",fe); print("FDM L2",de); print("FEM nodal",fn)
print("orders FEM L2",np.log2(fe[:-1]/fe[1:]).round(3)); print("orders nodal",np.log2(fn[:-1]/fn[1:]).round(3))
fig,axs=plt.subplots(1,3,figsize=(17,5.2))
ax=axs[0]; N=6; xn=np.linspace(0,1,N+1); xx=np.linspace(0,1,601)
sty=['-','--','-.',':','-','--']
for i in range(1,N):
    phi=np.clip(1-np.abs(xx-xn[i])*N,0,None)
    ax.plot(xx,phi,ls=sty[i-1],lw=2.2,color=bs.CYCLE[(i-1)%6])
    ax.text(xn[i],1.04,rf"$\phi_{i}$",ha='center',va='bottom',fontsize=17)
ax.plot(xn,np.zeros_like(xn),'ko',ms=6)
ax.set_xlabel("x"); ax.set_ylabel(r"$\phi_i(x)$"); ax.set_ylim(0,1.2); ax.set_xticks(xn); ax.set_xticklabels([f"{v:.2f}" for v in xn],fontsize=14)
ax=axs[1]; x,u=fem_p1(6)
ax.plot(xx,exact(xx),'-',color=bs.NAVY,lw=2.4,label=r"exact  sin($\pi$x)")
ax.plot(x,u,'--o',color=bs.RED,lw=2.2,ms=8,label="P1 Galerkin, N = 6")
ax.set_xlabel("x"); ax.set_ylabel("u(x)"); ax.legend(frameon=False,loc='lower center'); ax.set_ylim(-0.05,1.1)
ax=axs[2]
ax.loglog(hs,fe,'o-',color=bs.NAVY,lw=2.2,ms=9,label=r"FEM  $L^2$ error")
ax.loglog(hs,de,'s--',color=bs.RED,lw=2,ms=12,mfc='none',mew=2,label=r"FDM  $L^2$ error")
ax.loglog(hs,fn,'^-.',color=bs.GREEN,lw=2.2,ms=9,label="FEM nodal max error")
ax.loglog(hs,3*hs**2,':',color=bs.GRAY,lw=2); ax.text(hs[3],3*hs[3]**2*2.5,"slope 2",fontsize=15,color=bs.GRAY,ha='center')
ax.loglog(hs,0.02*hs**4,':',color=bs.GRAY,lw=2); ax.text(hs[4]*0.85,0.02*hs[4]**4*0.5,"slope 4",fontsize=15,color=bs.GRAY,ha="left",va="top")
ax.set_xlabel("h"); ax.set_ylabel("error"); ax.legend(frameon=False,loc='lower right',fontsize=14)
for a,l in zip(axs,'abc'): bs.panel_label(a,l)
fig.tight_layout(w_pad=2.5); fig.savefig('../ch02/figures/Fig2.4.2_fem.png')
