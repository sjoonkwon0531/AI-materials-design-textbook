import numpy as np
m=k=1.0; w2=k/m; x0,v0=1.0,0.0; dt=0.1; T=30.0     # professor's Wk6 setup
N=int(T/dt); energy=lambda x,v: 0.5*v**2+0.5*x**2   # exact E0=0.5

# explicit Euler
xe,ve=[x0],[v0]
for n in range(N):
    xe.append(xe[-1]+dt*ve[-1]); ve.append(ve[-1]-dt*w2*xe[-2])
# velocity-Verlet (Box 2.6)
xv,vv=x0,v0; a=-w2*xv; Xv=[xv]; Vv=[vv]
for n in range(N):
    xn=xv+vv*dt+0.5*a*dt*dt; an=-w2*xn; vv=vv+0.5*(a+an)*dt; xv=xn; a=an
    Xv.append(xv); Vv.append(vv)
