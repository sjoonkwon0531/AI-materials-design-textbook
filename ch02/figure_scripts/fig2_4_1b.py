"""Fig 2.4.1b: professor's Wk7 Poisson examples (1D exact FDM, 2D solution, block-tridiagonal sparsity)."""
import sys; sys.path.insert(0,'../../ch02/figure_scripts')
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
exec(open('../../ch02/figure_scripts/ex2_4_poisson.py').read().split('prev=None')[0].replace('print(','(lambda *a,**k:None)('))
fig,axs=plt.subplots(1,3,figsize=(17,5.8),gridspec_kw=dict(width_ratios=[1.1,0.7,0.95]))
# (a) 1D
ax=axs[0]; xx=np.linspace(0,1,300)
ax.plot(xx,ex1(xx),'-',color=bs.NAVY,lw=2.4,label="exact  $x^2/2-0.3x+2$")
x,u=poisson1d(0.2)
ax.plot(x[1:-1],u[1:-1],'o',color=bs.RED,ms=11,label="FDM unknowns $u_1..u_{N-1}$ (h = 0.2)")
ax.plot(x[[0,-1]],u[[0,-1]],'s',mfc='white',mec=bs.RED,mew=2,ms=11,label="boundary values $u_0$, $u_N$")
for i in range(len(x)):
    ax.annotate(f"$u_{{{i}}}$" if i<len(x)-1 else "$u_N$",(x[i],u[i]),xytext=(0,-26),textcoords='offset points',ha='center',fontsize=15)
ax.set_xlabel("x"); ax.set_ylabel("u(x)"); ax.set_ylim(1.8,2.3)
ax.legend(frameon=False,loc='upper left',fontsize=14)
# (b) 2D solution
ax=axs[1]
xg,yg,U,err=poisson2d(0.0125,return_A=True)[:4]
Uf=np.zeros((len(xg),len(yg))); Uf[1:-1,1:-1]=U; Uf[:,0]=ex2(xg,yg[0]); Uf[:,-1]=ex2(xg,yg[-1])
X,Y=np.meshgrid(xg,yg,indexing='ij')
cf=ax.contourf(X,Y,Uf*1e3,levels=21,cmap='RdBu_r')
ax.contour(X,Y,Uf,levels=[0],colors='k',linewidths=1.0,linestyles='--')
cb=fig.colorbar(cf,ax=ax,fraction=0.08,pad=0.04); cb.set_label(r"u  ($\times10^{-3}$)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect('equal'); ax.set_xticks([0,0.5,1])
# (c) sparsity
ax=axs[2]
_,_,_,_,A=poisson2d(0.25,return_A=True); ny=int(round(2/0.25))-1
ax.spy(A,markersize=7,color=bs.NAVY)
n=A.shape[0]
for k in range(1,n//ny):
    ax.axhline(k*ny-0.5,color=bs.GRAY,lw=0.8,ls=':'); ax.axvline(k*ny-0.5,color=bs.GRAY,lw=0.8,ls=':')
ax.text(ny*0.5-0.5,-1.2,"D",ha='center',fontsize=18,fontweight='bold')
ax.text(ny*1.5-0.5,-1.2,"C = I",ha='center',fontsize=18,fontweight='bold')
ax.text(ny*2.5-0.5,-1.2,"Z = 0",ha='center',fontsize=18,fontweight='bold')
ax.set_xlabel(f"h = 0.25:  {n}$\\times${n} matrix\n$N_y$ = {ny} interior points per column",labelpad=8)
ax.xaxis.set_label_position('bottom'); ax.tick_params(labelsize=14,top=False,labeltop=False,bottom=True,labelbottom=True)
for a,l in zip(axs,'abc'): bs.panel_label(a,l,y=1.06 if l=='c' else 1.02)
fig.tight_layout(w_pad=2)
fig.savefig('../../ch02/figures/Fig2.4.1b_poisson_fdm.png')
print(A.shape, A.nnz, "offsets", sorted(set((A.tocoo().col-A.tocoo().row).tolist())))
