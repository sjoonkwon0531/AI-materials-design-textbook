"""Worked Example (Wk7): y' = -2y + 3e^{-4t}, y(0)=1, RK4 convergence at T=5."""
import numpy as np
f=lambda t,y: -2*y+3*np.exp(-4*t)
ex=lambda t: 2.5*np.exp(-2*t)-1.5*np.exp(-4*t)
T=5.0; prev=None
for dt in [0.1,0.05,0.025,0.0125,0.001]:
    n=int(round(T/dt)); y=1.0; t=0.0
    for i in range(n):
        k1=f(t,y);k2=f(t+dt/2,y+dt/2*k1);k3=f(t+dt/2,y+dt/2*k2);k4=f(t+dt,y+dt*k3)
        y+=dt/6*(k1+2*k2+2*k3+k4); t=(i+1)*dt
    e=abs(y-ex(T)); print("%7g  %.6e  %.2e  %s"%(dt,y,e,"" if prev is None else "%.1fx"%(prev/e))); prev=e
for dt in [0.1,0.05,0.025]:
    n=int(round(T/dt)); y=1.0
    for i in range(n): y+=dt*f(i*dt,y)
    print("Euler",dt,abs(y-ex(T)))
