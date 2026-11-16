# numpy mirror of lab_4_1.jl (same matrices, same print order) -> numbers must match the chapter
import numpy as np, scipy.linalg as la
def H_(x,V):
    N=len(x); h=x[1]-x[0]
    D2=(np.diag(np.full(N,-2.0))+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1))/h**2
    return -0.5*D2+np.diag(V)
x=np.linspace(-10,10,1000); E,psi=la.eigh(H_(x,0.5*x**2))
for n in range(6): print(f"{n:2d}  {E[n]:9.5f}  {n+0.5:8.4f}  {abs(E[n]-(n+0.5)):.2e}")
print(np.round(psi[:,:4].T@psi[:,:4],4)+0.0)
xb=np.linspace(0,1,2000); Eb=la.eigvalsh(H_(xb,np.zeros(2000)))
for n in range(1,6):
    Eex=n**2*np.pi**2/2; print(f"{n:2d}  {Eb[n-1]:9.4f}  {Eex:9.4f}  {abs(Eb[n-1]-Eex)/Eex:.2e}")
s=""
for Ng in (100,200,400,800,1600,3200):
    xg=np.linspace(-10,10,Ng); Eg=la.eigvalsh(H_(xg,0.5*xg**2)); s+=f"{'N=' if Ng==100 else ''}{Ng}:{abs(Eg[0]-0.5):.2e}  "
print(s)
a=np.linspace(0.2,4.0,400); Ev=a/4+1/(4*a); i=np.argmin(Ev); print(f"{a[i]:.4f} {Ev[i]:.5f}")
