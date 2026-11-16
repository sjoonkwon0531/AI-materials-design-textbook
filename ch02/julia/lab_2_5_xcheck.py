import numpy as np
kap,Ta,L,T0,TL=0.01,20.0,10.0,40.0,200.0
rhs=lambda Y: np.array([Y[1],kap*(Y[0]-Ta)])
def shoot(s,dx=0.01,full=False):
    Y=np.array([T0,s]);p=[Y[0]]
    for _ in range(int(round(L/dx))):
        k1=rhs(Y);k2=rhs(Y+dx/2*k1);k3=rhs(Y+dx/2*k2);k4=rhs(Y+dx*k3);Y=Y+dx/6*(k1+2*k2+2*k3+k4);p.append(Y[0])
    return np.array(p) if full else Y[0]
m=np.sqrt(kap);c=np.linalg.solve([[1,1],[np.exp(m*L),np.exp(-m*L)]],[T0-Ta,TL-Ta]);Tex=lambda x:c[0]*np.exp(m*x)+c[1]*np.exp(-m*x)+Ta
I1,I2=0.,500.;fh=0
for it in range(1,51):
    s=0.5*(I1+I2);t=shoot(s);e=t-TL
    if fh==0 and abs(e)<1e-8: fh=it
    if it in (1,5,20,39): print("%4d  %12.6f  %11.6f  %+.2e"%(it,s,t,e))
    if t<TL: I1=s
    else: I2=s
print(fh,"%.6f"%s,"%.1e"%np.abs(shoot(s,full=True)-Tex(np.linspace(0,L,1001))).max())
g0=shoot(0.)-TL;g1=shoot(500.)-TL;ss=-g0*500/(g1-g0);print("%.6f %.1e"%(ss,shoot(ss)-TL))
