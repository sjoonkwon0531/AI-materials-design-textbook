import numpy as np
N=256; L=4*np.pi; eps=0.10; dt=0.05; S=2.0
k1=np.fft.fftfreq(N,d=L/N)*2*np.pi
KX,KY=np.meshgrid(k1,k1,indexing='ij'); K2=KX**2+KY**2; K4=K2**2
denom=1+dt*S*K2+dt*eps**2*K4; gain=1+dt*S*K2

c=0.1*(2*np.random.default_rng(1).random((N,N))-1)   # quench: small noise
chat=np.fft.fft2(c)
for n in range(8000):
    chat=(gain*chat - dt*K2*np.fft.fft2(c**3-c))/denom   # Box 2.22, one line
    c=np.real(np.fft.ifft2(chat))
