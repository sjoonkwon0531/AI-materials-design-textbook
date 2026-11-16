import numpy as np
from scipy.linalg import expm
lap=lambda N: (np.diag(-2*np.ones(N-1))+np.diag(np.ones(N-2),1)+np.diag(np.ones(N-2),-1))*N**2
lam_k=lambda N,k: 4*N**2*np.sin(k*np.pi/(2*N))**2
N=50;print("%.3f %.4f"%(lam_k(N,1),-np.linalg.eigvalsh(lap(N)).max()))
for n in [10,20,40,80,160]: print(n,"%.3f"%((np.pi**2-lam_k(n,1))*n**2))
h=1/N;x=np.linspace(0,1,N+1);xi=x[1:-1];T=0.2;u0=1-np.abs(2*xi-1);Ah=lap(N)*h**2;uref=expm(lap(N)*T)@u0;I=np.eye(N-1)
for lab,lam,s in [("F.4",.4,'f'),("F.52",.52,'f'),("BE",10.,'b'),("CN",10.,'c')]:
    dt=lam*h*h;n=int(round(T/dt));u=u0.copy()
    P=np.linalg.inv(I-lam*Ah) if s=='b' else (np.linalg.solve(I-.5*lam*Ah,I+.5*lam*Ah) if s=='c' else None)
    for _ in range(n): u=u+lam*(Ah@u) if s=='f' else P@u
    print(lab,n,"%.4g %.2e"%(np.abs(u).max(),np.abs(u-uref).max()))
print("%.4f"%np.abs(uref).max())
for n in [10,20,40,80,160,320]: print(n,"%.1f"%(lam_k(n,n-1)/lam_k(n,1)))
