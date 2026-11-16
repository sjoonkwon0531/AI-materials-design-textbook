"""Lab 2.7: 1D heat equation u_t = u_xx on [0,1], u(0)=u(1)=0, by Method of Lines.
Parameters (stated in text): alpha=1, N=50 intervals (h=0.02), IC = triangular hat 1-|2x-1|, t_final=0.2."""
import sys; sys.path.insert(0,'../../ch02/figure_scripts')
import numpy as np, scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh
import bookstyle as bs, matplotlib.pyplot as plt, logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def lap(N):  # interior (N-1)x(N-1) matrix tridiag(1,-2,1)/h^2
    h=1/N; m=N-1
    return sp.diags([np.ones(m-1),-2*np.ones(m),np.ones(m-1)],[-1,0,1],format='csc')/h**2
def lam_k(N,k): h=1/N; return 4/h**2*np.sin(k*np.pi*h/2)**2   # |eigenvalue| k

# (1) smallest eigenmode
N=50
print("(1) N=50: lambda1 = %.3f  vs pi^2 = %.3f"%(lam_k(N,1),np.pi**2))
ev=-eigsh(lap(N),k=1,sigma=0,which='LM')[0][0]; print("    (numeric eigsh check %.4f)"%ev)
# (2) eigenvalue error / h^2
print("(2) (pi^2 - lambda1)/h^2:")
for n in [10,20,40,80,160]:
    print("   N=%4d  %.4f"%(n,(np.pi**2-lam_k(n,1))*n**2))
print("    limit pi^4/12 = %.4f"%(np.pi**4/12))
# solution-error check with sin IC (2nd order of MOL solution)
for n in [10,20,40,80]:
    x=np.linspace(0,1,n+1)[1:-1]; T=0.1
    s=solve_ivp(lambda t,u: lap(n)@u,[0,T],np.sin(np.pi*x),method='BDF',jac=lap(n),rtol=1e-11,atol=1e-13)
    e=np.abs(s.y[:,-1]-np.exp(-np.pi**2*T)*np.sin(np.pi*x)).max(); print("   sinIC T=0.1 N=%d err/h^2=%.4f"%(n,e*n**2))

# (3) time stepping on N=50, hat IC, T=0.2
h=1/N; x=np.linspace(0,1,N+1); xi=x[1:-1]; m=N-1; T=0.2
u0=1-np.abs(2*xi-1)
A=(lap(N)*h**2).toarray(); I=np.eye(m)
def march(lam,scheme):
    dt=lam*h*h; n=int(round(T/dt)); u=u0.copy(); hist=[np.abs(u).max()]
    if scheme=='be': P=np.linalg.inv(I-lam*A)
    if scheme=='cn': P=np.linalg.solve(I-0.5*lam*A,I+0.5*lam*A)
    for _ in range(n):
        u = u+lam*(A@u) if scheme=='ftcs' else P@u
        hist.append(np.abs(u).max())
    return np.arange(n+1)*dt,np.array(hist),u
ref=solve_ivp(lambda t,u: lap(N)@u,[0,T],u0,method='BDF',jac=lap(N),rtol=1e-12,atol=1e-14).y[:,-1]
res={}
for lab,lam,sch in [('FTCS 0.40',0.40,'ftcs'),('FTCS 0.52',0.52,'ftcs'),('BE 10',10,'be'),('CN 10',10,'cn')]:
    t,hh,u=march(lam,sch); res[lab]=(t,hh)
    print("(3) %-10s steps=%4d  max|u(T)|=%.4g   max|u-u_MOL|=%.2e"%(lab,len(t)-1,hh[-1],np.abs(u-ref).max()))
print("    reference MOL(BDF) max|u(T)| = %.4f"%np.abs(ref).max())
# CN 2nd-order check at fixed h
for lam in [10,5,2.5]:
    _,_,u=march(lam,'cn'); _,_,ub=march(lam,'be')
    print("    lam=%5.2f CN err %.2e  BE err %.2e"%(lam,np.abs(u-ref).max(),np.abs(ub-ref).max()))

# ---------------- figure ----------------
fig,axs=plt.subplots(1,3,figsize=(17,5.2))
ax=axs[0]
sol=solve_ivp(lambda t,u: lap(N)@u,[0,T],u0,method='BDF',jac=lap(N),rtol=1e-10,atol=1e-12,
              t_eval=[0,0.01,0.03,0.08,0.2])
sty=['-','--','-.',':','-']; mk=[None,None,None,None,'o']
for j,tt in enumerate(sol.t):
    ax.plot(x,np.r_[0,sol.y[:,j],0],ls=sty[j],marker=mk[j],markevery=5,ms=5,lw=2.2,
            color=bs.CYCLE[j],label=f"t = {tt:g}")
ax.set_xlabel("x"); ax.set_ylabel("u(x, t)"); ax.legend(frameon=False,loc='upper right',fontsize=14)
ax.set_ylim(0,1.12)
ax=axs[1]
st={'FTCS 0.40':dict(ls='-',color=bs.NAVY,label=r"FTCS, $\lambda$ = 0.40"),
    'FTCS 0.52':dict(ls='--',color=bs.RED,label=r"FTCS, $\lambda$ = 0.52"),
    'BE 10':dict(ls='none',marker='s',ms=7,markevery=(0,6),color=bs.GREEN,label=r"Backward Euler, $\lambda$ = 10"),
    'CN 10':dict(ls='none',marker='^',ms=7,markevery=(3,6),color=bs.GOLD,label=r"Crank-Nicolson, $\lambda$ = 10")}
for k,(t,hh) in res.items():
    ax.semilogy(t,hh,lw=2.2,**st[k])
ax.set_xlabel("t"); ax.set_ylabel(r"max$_i$ |u$_i$|"); ax.set_ylim(1e-2,1e32); ax.set_yticks([1e0,1e10,1e20,1e30])
ax.text(0.1,3e-3*1e4,'stable runs',ha='center',va='bottom',fontsize=15,color=bs.GRAY)
ax.legend(frameon=False,loc='upper left',fontsize=14)
ax=axs[2]
Ns=np.array([10,20,40,80,160,320]); S=[lam_k(n,n-1)/lam_k(n,1) for n in Ns]
ax.loglog(Ns,S,'o',color=bs.NAVY,ms=10,label="stiffness ratio")
ax.loglog(Ns,(2/np.pi)**2*Ns**2,'--',color=bs.RED,lw=2,label=r"$(2/\pi)^2 N^2$")
ax.set_xlabel("N (grid intervals)"); ax.set_ylabel(r"$|\lambda_{max}|\,/\,|\lambda_{min}|$")
ax.legend(frameon=False,loc='upper left')
for a,l in zip(axs,'abc'): bs.panel_label(a,l)
fig.tight_layout(w_pad=2.5)
fig.savefig('../../ch02/figures/Fig2.4.1_heat_mol.png')
print("stiffness", dict(zip(Ns,np.round(S,1))))
