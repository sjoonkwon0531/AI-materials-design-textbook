"""Lab 2.13: upwind advection of a square wave, CFL check.
Parameters (stated in text): periodic x in [0,1), N=200 cells (dx=0.005),
square wave u=1 on [0.1,0.3), 0 elsewhere, a=1, 300 steps."""
import numpy as np
N=200; dx=1.0/N; x=np.arange(N)*dx
u0=((x>=0.1)&(x<0.3)).astype(float)
def upwind(u,nu,steps):
    u=u.copy()
    for n in range(steps): u=u-nu*(u-np.roll(u,1))
    return u
res={}
for nu in (0.8,1.0,1.3):
    u=upwind(u0,nu,300); res[nu]=u
    print(f"CFL nu={nu}: max|u| = {np.abs(u).max():.6g}")
# exactness at nu=1: shift by 300 cells
print("nu=1 exact shift:", np.array_equal(res[1.0], np.roll(u0,300)))
