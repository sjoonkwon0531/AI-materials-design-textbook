"""Lab 2.2 / Fig 2.2.4: mass-spring energy with explicit Euler, implicit Euler, RK4, velocity-Verlet."""
import sys; sys.path.insert(0,'../../ch02/figure_scripts')
import numpy as np, bookstyle as bs, matplotlib.pyplot as plt
w2=1.0; x0,v0=1.0,0.0; dt=0.1
E=lambda x,v: 0.5*v**2+0.5*w2*x**2
def run(T):
    N=int(round(T/dt))
    out={}
    # explicit Euler
    X=np.empty(N+1);V=np.empty(N+1);X[0],V[0]=x0,v0
    for n in range(N): X[n+1]=X[n]+dt*V[n]; V[n+1]=V[n]-dt*w2*X[n]
    out['fe']=(X,V)
    # implicit Euler
    M=np.linalg.inv(np.array([[1,-dt],[dt*w2,1]]))
    X=np.empty(N+1);V=np.empty(N+1);X[0],V[0]=x0,v0
    for n in range(N): X[n+1],V[n+1]=M@np.array([X[n],V[n]])
    out['be']=(X,V)
    # RK4
    f=lambda y: np.array([y[1],-w2*y[0]])
    X=np.empty(N+1);V=np.empty(N+1);X[0],V[0]=x0,v0; y=np.array([x0,v0])
    for n in range(N):
        k1=f(y);k2=f(y+dt/2*k1);k3=f(y+dt/2*k2);k4=f(y+dt*k3); y=y+dt/6*(k1+2*k2+2*k3+k4)
        X[n+1],V[n+1]=y
    out['rk4']=(X,V)
    # velocity Verlet
    X=np.empty(N+1);V=np.empty(N+1);X[0],V[0]=x0,v0; x,v=x0,v0; a=-w2*x
    for n in range(N):
        xn=x+v*dt+0.5*a*dt*dt; an=-w2*xn; v=v+0.5*(a+an)*dt; x=xn; a=an; X[n+1],V[n+1]=x,v
    out['vv']=(X,V)
    return out
o=run(30.0); t=np.arange(len(o['fe'][0]))*dt
print("T=30, periods=%.2f"%(30/(2*np.pi)))
for k in o:
    X,V=o[k]; e=E(X,V); print(k,"E(30)=%.6f ratio=%.4f max|x|=%.3f Emin=%.5f Emax=%.5f"%(e[-1],e[-1]/0.5,np.abs(X).max(),e.min(),e.max()))
ol=run(2000.0)
print("T=2000 periods=%.2f steps=%d"%(2000/(2*np.pi),int(round(2000/dt))))
for k in ['rk4','vv']:
    X,V=ol[k]; e=E(X,V); print(k,"E_end=%.6f drift=%.4f%% min=%.6f max=%.6f band=%.3e"%(e[-1],100*(e[-1]/0.5-1),e.min(),e.max(),e.max()-e.min()))
g=1+(1j*dt)+(1j*dt)**2/2+(1j*dt)**3/6+(1j*dt)**4/24
print("analytic |g|^2=",abs(g)**2," E(2000)=",0.5*abs(g)**(2*20000))
print("area factor Euler",1+dt**2,(1+dt**2)**300)

# figure
fig,axs=plt.subplots(1,3,figsize=(17,5.6))
st={'fe':dict(c=bs.RED,ls='-',lw=2.2,label='Explicit Euler'),
    'be':dict(c=bs.BLUE,ls='--',lw=2.2,label='Implicit Euler'),
    'rk4':dict(c=bs.GOLD,ls='none',marker='s',ms=7,markevery=(12,25),label='RK4'),
    'vv':dict(c=bs.GREEN,ls='-',lw=2.2,marker='o',ms=6,markevery=25,label='Velocity Verlet')}
ex=dict(c=bs.NAVY,lw=1.6,ls=':',label=r'Exact $\cos t$')
ax=axs[0]
for k in ['fe','vv']: ax.plot(t,o[k][0],**st[k])
ax.plot(t,np.cos(t),**ex)
ax.set_xlabel('t'); ax.set_ylabel('x(t)')
ax=axs[1]
for k in ['fe','be','vv','rk4']:
    X,V=o[k]; ax.semilogy(t,E(X,V),**st[k])
ax.set_xlabel('t'); ax.set_ylabel('Energy E(t)')
ax=axs[2]
for k in ['fe','be','vv']:
    X,V=o[k]; ax.plot(X,V,**{**st[k],'lw':1.5,'marker':None})
ax.set_xlabel('x'); ax.set_ylabel('v'); ax.set_aspect('equal'); ax.set_xlim(-4.6,4.6); ax.set_ylim(-4.6,4.6)
for a,l in zip(axs,'abc'): bs.panel_label(a,l)
from matplotlib.lines import Line2D
h=[Line2D([],[],**{kk:vv for kk,vv in st[k].items() if kk!='markevery'}) for k in ['fe','be','rk4','vv']]+[Line2D([],[],**ex)]
fig.legend(handles=h,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(w_pad=2.5,rect=(0,0.09,1,1))
fig.savefig('../../ch02/figures/Fig2.2.4_energy_drift.png')
