"""Lab 2.5 / Fig 2.2.7: shooting method for T''=kap(T-Ta), T(0)=40, T(10)=200 (code block b02 + diagnostics)."""
import sys; sys.path.insert(0,'../ch02/figures')
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
kap, Ta, L, T0, TL = 0.01, 20.0, 10.0, 40.0, 200.0
rhs = lambda Y: np.array([Y[1], kap*(Y[0]-Ta)])
def shoot(slope, dx=0.01, full=False):
    Y=np.array([T0, slope]); out=[Y[0]]
    for _ in range(int(round(L/dx))):
        k1=rhs(Y); k2=rhs(Y+dx/2*k1); k3=rhs(Y+dx/2*k2); k4=rhs(Y+dx*k3)
        Y=Y+dx/6*(k1+2*k2+2*k3+k4); out.append(Y[0])
    return np.array(out) if full else Y[0]
print("int(L/dx) =",int(L/0.01),"(int(10/0.01) in float:",L/0.01,")")
# analytic
m=np.sqrt(kap); A=np.array([[1,1],[np.exp(m*L),np.exp(-m*L)]]); c=np.linalg.solve(A,[T0-Ta,TL-Ta])
Tex=lambda x: c[0]*np.exp(m*x)+c[1]*np.exp(-m*x)+Ta
print("analytic C1=%.4f C2=%.4f T'(0)=%.7f"%(c[0],c[1],m*(c[0]-c[1])))
I1,I2=0.0,500.0; hist=[]
for it in range(1,51):
    s=0.5*(I1+I2); TLs=shoot(s); hist.append((it,s,TLs,TLs-TL))
    if TLs<TL: I1=s
    else: I2=s
for it,s,t,e in hist:
    if it in (1,5,20,38,39,40) or it<=3: print("%3d %.7f %.6f %+.2e"%(it,s,t,e))
first=[it for it,s,t,e in hist if abs(e)<1e-8]; print("first iter |err|<1e-8:",first[0] if first else None)
first7=[it for it,s,t,e in hist if abs(e)<1e-7]; print("first iter |err|<1e-7:",first7[0])
# stopping on bracket width: I2-I1 = 500/2^n
for tol in [1e-8,1e-9]:
    n=int(np.ceil(np.log2(500/tol))); print("bracket width < %g after %d halvings"%(tol,n))
sconv=hist[-1][1]; print("converged slope %.7f"%sconv)
x=np.linspace(0,L,int(round(L/0.01))+1); prof=shoot(sconv,full=True)
print("max profile err %.2e"%np.abs(prof-Tex(x)).max())
g0=shoot(0)-TL; g1=shoot(500)-TL; ssec=0-g0*(500-0)/(g1-g0); print("secant slope %.7f  resid %.1e"%(ssec,shoot(ssec)-TL))
# figure
fig,axs=plt.subplots(1,2,figsize=(14,5.4))
ax=axs[0]
for s,ls,lab in [(8,':','s = 8 (too low)'),(10,'-.','s = 10'),(15,'--','s = 15 (too high)')]:
    ax.plot(x,shoot(s,full=True),color=bs.GRAY,lw=2,ls=ls,label=lab)
ax.plot(x,prof,color=bs.GREEN,lw=3,label='s = 12.69 (hit)')
ax.plot([L],[TL],'*',ms=20,color=bs.GOLD,mec='k',zorder=5,label='Target T(L) = 200')
ax.plot([0],[T0],'o',ms=9,color='k')
ax.set_xlabel('x'); ax.set_ylabel('T(x)'); ax.legend(loc='upper left',frameon=False,fontsize=14)
ax=axs[1]
its=[h[0] for h in hist[:45]]; er=[abs(h[3]) for h in hist[:45]]
ax.semilogy(its,er,'o-',color=bs.NAVY,ms=5,lw=1.6,label='Bisection')
ax.semilogy([1],[abs(shoot(ssec)-TL)+1e-16],'*',ms=20,color=bs.GREEN,mec='k',label='Secant (1 step)')
ax.axhline(1e-8,color=bs.RED,ls='--',lw=1.5); ax.text(1,2.5e-8,'tol = $10^{-8}$',ha='left',color=bs.RED,fontsize=15)
ax.annotate('first below tol:\niteration 39',xy=(39,er[38]),xytext=(27,3e-12),fontsize=15,arrowprops=dict(arrowstyle='->'))
ax.set_xlabel('Iteration'); ax.set_ylabel('|T(L) - 200|'); ax.legend(loc='upper right',frameon=False)
for a,l in zip(axs,'ab'): bs.panel_label(a,l)
fig.tight_layout(w_pad=3)
fig.savefig('../ch02/figures/Fig2.2.7_shooting.png')
