"""Fig 2.4.3: first-order upwind FVM, u_t+u_x=0, N=200, CFL=0.8, one period."""
import sys; sys.path.insert(0,'../../ch02/figure_scripts')
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
exec(open('../../ch02/labs/b04_L1335.py').read())
xc,u0,u,mass=run_fvm(); drift=np.abs(mass-mass[0])/mass[0]
print("max drift %.3e, nonzero steps %d/%d"%(drift.max(),(drift>0).sum(),len(drift)))
fig,axs=plt.subplots(1,2,figsize=(14,5.2))
ax=axs[0]
ax.step(xc,u0,where='mid',ls='--',color=bs.NAVY,lw=2.2,label="exact (initial = t = 1)")
ax.plot(xc,u,'-',color=bs.RED,lw=2.4,label="upwind FVM, t = 1")
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.set_ylim(-0.05,1.3); ax.legend(frameon=False,loc='upper right')
ax=axs[1]; steps=np.arange(len(mass))
nz=drift>0
ax.semilogy(steps[nz],drift[nz],'o',ms=4,color=bs.NAVY,label="relative mass drift")
ax.axhline(np.finfo(float).eps,ls='--',color=bs.RED,lw=2,label=r"machine epsilon 2.2$\times10^{-16}$")
ax.set_ylim(1e-18,1e-13); ax.set_xlabel("time step"); ax.set_ylabel(r"|M(t) - M(0)| / M(0)")
ax.legend(frameon=False,loc='upper left')
for a,l in zip(axs,'ab'): bs.panel_label(a,l)
fig.tight_layout(w_pad=3); fig.savefig('../../ch02/figures/Fig2.4.3_fvm.png')
