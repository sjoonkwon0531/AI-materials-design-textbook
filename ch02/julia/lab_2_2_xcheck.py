# numpy mirror of lab_2_2.jl
import numpy as np
w2=1.0;x0,v0=1.0,0.0;dt=0.1;E=lambda x,v:0.5*v**2+0.5*w2*x**2
def run(T):
    N=int(round(T/dt));o={}
    X=np.zeros(N+1);V=np.zeros(N+1);X[0],V[0]=x0,v0
    for n in range(N): X[n+1]=X[n]+dt*V[n];V[n+1]=V[n]-dt*w2*X[n]
    o['FE']=(X,V);M=np.linalg.inv(np.array([[1,-dt],[dt*w2,1]]))
    X=np.zeros(N+1);V=np.zeros(N+1);X[0],V[0]=x0,v0
    for n in range(N): X[n+1],V[n+1]=M@np.array([X[n],V[n]])
    o['BE']=(X,V);f=lambda y:np.array([y[1],-w2*y[0]])
    X=np.zeros(N+1);V=np.zeros(N+1);X[0],V[0]=x0,v0;y=np.array([x0,v0])
    for n in range(N):
        k1=f(y);k2=f(y+dt/2*k1);k3=f(y+dt/2*k2);k4=f(y+dt*k3);y=y+dt/6*(k1+2*k2+2*k3+k4);X[n+1],V[n+1]=y
    o['RK4']=(X,V)
    X=np.zeros(N+1);V=np.zeros(N+1);X[0],V[0]=x0,v0;x,v=x0,v0;a=-w2*x
    for n in range(N):
        xn=x+v*dt+0.5*a*dt**2;an=-w2*xn;v=v+0.5*(a+an)*dt;x=xn;a=an;X[n+1],V[n+1]=x,v
    o['VV']=(X,V);return o
o=run(30.0)
for k,(X,V) in o.items():
    e=E(X,V);print(k,"E(30)=%.6f E/E0=%.4f max|x|=%.3f [%.4f,%.4f]"%(e[-1],e[-1]/.5,np.abs(X).max(),e.min(),e.max()))
o=run(2000.0)
for k in ['RK4','VV']:
    X,V=o[k];e=E(X,V);print(k,"%.6f %.4f%% [%.6f,%.6f] %.3e"%(e[-1],100*(e[-1]/.5-1),e.min(),e.max(),e.max()-e.min()))
