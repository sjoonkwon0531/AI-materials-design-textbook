import sys; sys.path.insert(0,'../ch02/figures'); import bookstyle as bs
import numpy as np, matplotlib.pyplot as plt, logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
d=np.load('lab2_14_data.npz'); L=float(d['L'])
keys=[('c0','t = 0'),('c10','t = 1'),('c50','t = 5'),('c400','t = 40'),('c4000','t = 400')]
fig,axs=plt.subplots(1,5,figsize=(20,4.9))
for ax,(k,lab),pl in zip(axs,keys,'abcde'):
    c=d[k]; vmax=1.0 if k!='c0' else 0.1
    im=ax.imshow(c.T,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax,extent=[0,L,0,L],interpolation='bilinear')
    if k!='c0': ax.contour(np.linspace(0,L,c.shape[0]),np.linspace(0,L,c.shape[1]),c.T,levels=[0],colors='white',linewidths=1.0)
    ax.set_xticks([]); ax.set_yticks([]); bs.panel_label(ax,pl,x=0.0,y=1.02)
    for s in ax.spines.values(): s.set_visible(True)
    ax.text(0.5,-0.08,lab,transform=ax.transAxes,ha='center',va='top',fontsize=19)
cb=fig.colorbar(im,ax=axs,fraction=0.012,pad=0.01); cb.set_label("c"); cb.set_ticks([-1,0,1])
ax0=axs[0]; ax0.text(1.0,1.02,"color range ±0.1",transform=ax0.transAxes,ha='right',va='bottom',fontsize=14)
fig.savefig("Fig2.6.2_spinodal.png")
