# numpy mirror of lab_2_14.jl's diagnostics (lstsq slope instead of polyfit; same everything else)
import numpy as np
N=256;L=4*np.pi;eps=0.10;dt=0.05;S=2.0;NSTEP=8000;dx=L/N
k1=np.fft.fftfreq(N,d=L/N)*2*np.pi;KX,KY=np.meshgrid(k1,k1,indexing='ij');K2=KX**2+KY**2;K4=K2**2
denom=1+dt*S*K2+dt*eps**2*K4;gain=1+dt*S*K2;kmag=np.sqrt(K2)
def energy(c):
    ch=np.fft.fft2(c);cx=np.real(np.fft.ifft2(1j*KX*ch));cy=np.real(np.fft.ifft2(1j*KY*ch))
    return np.sum(0.25*(c**2-1)**2+0.5*eps**2*(cx**2+cy**2))*dx*dx
def length(c):
    Sk=np.abs(np.fft.fft2(c-c.mean()))**2;Sk[0,0]=0;return 2*np.pi*Sk.sum()/np.sum(kmag*Sk)
c=0.1*(2*np.random.default_rng(1).random((N,N))-1);m0=c.mean();chat=np.fft.fft2(c)
t=[0.];F=[energy(c)];Lt=[length(c)];M=0.
for n in range(1,NSTEP+1):
    chat=(gain*chat-dt*K2*np.fft.fft2(c**3-c))/denom;c=np.real(np.fft.ifft2(chat));M=max(M,abs(c.mean()-m0))
    if n%20==0: t.append(n*dt);F.append(energy(c));Lt.append(length(c))
t=np.array(t);F=np.array(F);Lt=np.array(Lt);dF=np.diff(F)
print("[1] %.2e"%M);print("[2] %.3g -> %.3g, %d / %d"%(F[0],F[-1],(dF>0).sum(),len(dF)))
m=(t>=50)&(t<=400);X=np.c_[np.ones(m.sum()),np.log(t[m])];print("[3] %.2f"%np.linalg.lstsq(X,np.log(Lt[m]),rcond=None)[0][1])
