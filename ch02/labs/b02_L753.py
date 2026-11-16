import numpy as np
kap, Ta, L, T0, TL = 0.01, 20.0, 10.0, 40.0, 200.0
rhs = lambda Y: np.array([Y[1], kap*(Y[0]-Ta)])           # [T', T'']
def shoot(slope, dx=0.01):                                 # RK4 integrate IVP
    Y=np.array([T0, slope])
    for _ in range(int(L/dx)):
        k1=rhs(Y); k2=rhs(Y+dx/2*k1); k3=rhs(Y+dx/2*k2); k4=rhs(Y+dx*k3)
        Y=Y+dx/6*(k1+2*k2+2*k3+k4)
    return Y[0]                                            # T(L) for this slope
I1,I2=0.0,500.0                                            # bisection on slope (§1.6)
for _ in range(50):
    s=0.5*(I1+I2)
    if shoot(s) < TL: I1=s
    else: I2=s
