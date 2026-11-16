import numpy as np
def run_fvm(N=200, a=1.0, CFL=0.8, tfinal=1.0):
    dx = 1.0/N; xc = (np.arange(N)+0.5)*dx
    u  = np.where((xc>0.3)&(xc<0.6), 1.0, 0.0)   # 사각파 초기조건
    u0 = u.copy(); dt = CFL*dx/a; nu = a*dt/dx
    mass = [u.sum()*dx]; t = 0.0
    while t < tfinal-1e-12:
        if t+dt > tfinal: dt = tfinal-t; nu = a*dt/dx
        u = u - nu*(u - np.roll(u,1))            # 풍상, 주기경계
        t += dt; mass.append(u.sum()*dx)
    return xc, u0, u, np.array(mass)
