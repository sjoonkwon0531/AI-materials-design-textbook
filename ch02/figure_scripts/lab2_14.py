"""Lab 2.14: 2D Cahn-Hilliard, stabilized semi-implicit Fourier spectral.
N=256, L=4*pi, eps=0.1, dt=0.05, S=2, seed=1, 8000 steps (T=400).
Saves diagnostics to lab2_14_data.npz for Fig 2.6.2 / 2.6.3."""
import numpy as np
N=256; L=4*np.pi; eps=0.10; dt=0.05; S=2.0; NSTEP=8000
dx=L/N
k1=np.fft.fftfreq(N,d=L/N)*2*np.pi
KX,KY=np.meshgrid(k1,k1,indexing='ij'); K2=KX**2+KY**2; K4=K2**2
denom=1+dt*S*K2+dt*eps**2*K4; gain=1+dt*S*K2

def energy(c):
    ch=np.fft.fft2(c)
    cx=np.real(np.fft.ifft2(1j*KX*ch)); cy=np.real(np.fft.ifft2(1j*KY*ch))
    return np.sum(0.25*(c**2-1)**2 + 0.5*eps**2*(cx**2+cy**2))*dx*dx

kmag=np.sqrt(K2)
def length(c):
    ch=np.fft.fft2(c-c.mean()); Sk=np.abs(ch)**2; Sk[0,0]=0
    return 2*np.pi*np.sum(Sk)/np.sum(kmag*Sk)

c=0.1*(2*np.random.default_rng(1).random((N,N))-1)
mean0=c.mean()
chat=np.fft.fft2(c)
t=[0.0]; F=[energy(c)]; M=[0.0]; Lt=[length(c)]
snaps={0.0:c.copy()}; Mmax=0.0
snap_steps={20:1.0,100:5.0,800:40.0,8000:400.0}
for n in range(1,NSTEP+1):
    chat=(gain*chat - dt*K2*np.fft.fft2(c**3-c))/denom
    c=np.real(np.fft.ifft2(chat))
    Mmax=max(Mmax,abs(c.mean()-mean0))          # mass drift checked every step
    if n%20==0:
        t.append(n*dt); F.append(energy(c)); M.append(abs(c.mean()-mean0)); Lt.append(length(c))
    if n in snap_steps: snaps[snap_steps[n]]=c.copy()
t=np.array(t); F=np.array(F); M=np.array(M); Lt=np.array(Lt)
dF=np.diff(F)
print("samples",len(t))
print(f"[1] MASS: max|mean(t)-mean0| = {Mmax:.3g} (every step); {M.max():.3g} (every 20 steps)")
print(f"[2] F: {F[0]:.4g} -> {F[-1]:.4g}; after first sample F(1.0)={F[1]:.4g}; increases = {(dF>0).sum()} / {len(dF)}; max dF={dF.max():.3g}")
FIT=(50,400)
m=(t>=FIT[0])&(t<=FIT[1]); slope=np.polyfit(np.log(t[m]),np.log(Lt[m]),1)[0]
print(f'[3] COARSENING slope {FIT}: {slope:.2f}')
for lo in (20,40,50,100):
    m=t>=lo; p=np.polyfit(np.log(t[m]),np.log(Lt[m]),1)
    print(f"[3] slope t>={lo}: {p[0]:.3f}")
print("L(t) at 1,5,40,400:", [round(Lt[np.argmin(abs(t-s))],3) for s in (1,5,40,400)])
print("c range final", c.min(), c.max())
# stability numbers
kmax=np.pi/dx
print("dx",dx,"kmax",kmax, "max|k| on grid", np.sqrt(K2).max())
print("explicit limit 1/(eps^2 kmax^4) =",1/(eps**2*kmax**4), " steps to T=400:",400/(1/(eps**2*kmax**4)))
print("dx^4/(pi^4 eps^2) =", dx**4/(np.pi**4*eps**2))
# including the -k^2 part: sigma(k)=k^2-eps^2k^4 ; explicit Euler needs dt<=2/|sigma| for decaying modes
sig=K2-eps**2*K4; print("explicit Euler bound 2/max|sigma| =",2/np.abs(sig).max())
print("ratio dt/explicit =",0.05/(1/(eps**2*kmax**4)), 0.05/(2/np.abs(sig).max()))
kstar=1/(eps*np.sqrt(2)); print("k*",kstar,"k*L/2pi",kstar*L/(2*np.pi), "lambda*",2*np.pi/kstar)
np.savez("lab2_14_data.npz",t=t,F=F,M=M,Lt=Lt,**{f"c{int(k*10)}":v for k,v in snaps.items()},L=L)
