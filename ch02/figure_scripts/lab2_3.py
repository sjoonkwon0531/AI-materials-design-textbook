"""Lab 2.3 / Fig 2.2.5: stiff mass-conserving network, explicit Euler blow-up vs BDF.
Network (chosen so that eig = {-1001.5, -1.5, 0}):  X1 -> X2 (k=1001.5, fast),  X2 <-> X3 (k=0.5, k_-=1.0)."""
import sys; sys.path.insert(0,'../../ch02/figure_scripts')
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
A=np.array([[-1001.5,0.0,0.0],[1001.5,-0.5,1.0],[0.0,0.5,-1.0]])
x0=np.array([1.0,0.0,0.0]); T=8.0
lam=np.sort(np.linalg.eigvals(A).real); print("colsum",A.sum(0),"eig",lam,"S=",abs(lam[0])/abs(lam[1]),"2/|lmax|=",2/abs(lam[0]))
exact=lambda t: expm(A*t)@x0
def fe(dt):
    n=int(round(T/dt)); X=np.empty((n+1,3)); X[0]=x0
    with np.errstate(all='ignore'):
        for i in range(n): X[i+1]=X[i]+dt*(A@X[i])
    return np.arange(n+1)*dt,X
for dt in [0.0019,0.0020,0.0021,0.0022]:
    t,X=fe(dt); m=np.abs(X[-1]).max(); print("FE dt=%.4f max|x(T)|=%.3e"%(dt,m))
rtol,atol=1e-6,1e-9
res={}
for meth in ['RK45','BDF','Radau']:
    kw={}
    s=solve_ivp(lambda t,x: A@x,(0,T),x0,method=meth,rtol=rtol,atol=atol,**kw)
    err=np.abs(s.y[:,-1]-exact(T)).max()
    errall=max(np.abs(s.y[:,i]-exact(s.t[i])).max() for i in range(len(s.t)))
    res[meth]=s
    print(meth,"nfev=",s.nfev,"njev=",s.njev,"nlu=",s.nlu,"steps=",len(s.t)-1,"err(T)=%.1e maxerr=%.1e"%(err,errall))
print("ratio RK45/BDF nfev=%.1f"%(res['RK45'].nfev/res['BDF'].nfev))
# figure
fig,axs=plt.subplots(1,2,figsize=(14,5.4))
ax=axs[0]
for dt,c,ls,lab in [(0.0019,bs.NAVY,'--','stable'),(0.0022,bs.RED,'-','unstable')]:
    t,X=fe(dt); nrm=np.abs(X).max(1)
    ax.semilogy(t,nrm,color=c,lw=2.4,ls=ls,label=r'$\Delta t=%.4f$ (%s)'%(dt,lab))
ax.axvline(0,color='none')
ax.set_ylim(1e-2,1e305); ax.set_yticks([1e0,1e100,1e200,1e300])
ax.set_xlabel('t'); ax.set_ylabel(r'max$_i\,|X_i(t)|$, explicit Euler'); ax.legend(loc='upper left',frameon=False)
ax.text(0.97,0.06,r'limit $2/|\lambda_{\max}|=0.001997$',transform=ax.transAxes,ha='right',fontsize=15)
ax=axs[1]
tt=np.logspace(-5,np.log10(T),500); Ex=np.array([exact(q) for q in tt])
for i in range(3): ax.plot(tt,Ex[:,i],color=bs.GRAY,lw=1.4)
s=res['BDF']; mk=['o','s','^']; cols=[bs.NAVY,bs.RED,bs.BLUE]
for i,lab in enumerate([r'$X_1$',r'$X_2$',r'$X_3$']):
    ax.plot(s.t[1:],s.y[i,1:],ls='none',marker=mk[i],ms=6,mfc='none',mew=1.5,color=cols[i],label=lab+' (BDF)')
ax.plot([],[],color=bs.GRAY,lw=1.4,label=r'Exact $e^{At}X_0$')
ax.set_xscale('log'); ax.set_xlim(1e-5,T)
ax.set_xlabel('t'); ax.set_ylabel('Concentration')
ax.set_ylim(-0.03,1.42); ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.legend(loc='upper left',frameon=False,fontsize=14,ncol=2)
for a,l in zip(axs,'ab'): bs.panel_label(a,l)
fig.tight_layout(w_pad=3)
fig.savefig('../../ch02/figures/Fig2.2.5_stiffness.png')
