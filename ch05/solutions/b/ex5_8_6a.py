"""Ex 5.8.6(a) and the energy balance used to read the film runs of ex5_8_6.py.
F(x)/F_min = (x^2 + 2/x)/3 ; SST free energy per chain for the N=10 DPD model; field energy gain per chain
of registered lamellae ~ N h sqrt(S/n) (S/n values copied from the ex5_8_6.py outputs)."""
import numpy as np
A_, B_ = 1.0, 2.0                        # any A, B: D0^3 = B/(2A)
D0 = (B_/(2*A_))**(1/3); Fmin = A_*D0**2 + B_/D0
for x in (0.9, 1.1):
    F = A_*(x*D0)**2 + B_/(x*D0)
    print(f"x={x}: F/Fmin numeric {F/Fmin:.5f}, formula {(x*x+2/x)/3:.5f}")
N, a, chi = 10, 0.990, 0.286*15
D0 = (32*a**3*N**2*np.sqrt(chi/6)/np.pi**2)**(1/3); Fmin = 3*np.pi**2*D0**2/(32*N*a**2)
print(f"SST N=10: D0={D0:.3f}, F_min={Fmin:.3f} kT/chain")
runs = {0.6: 0.148, 0.7: 0.349, 0.8: 0.378, 0.9: 0.384, 1.0: 0.388, 1.1: 0.387, 1.2: 0.380, 1.3: 0.378, 1.5: 0.256, 1.7: 0.059}
print(" x    cost (kT/chain)   field gain N h sqrt(S/n) at h=0.2 (kT/chain)   gain at h=0.05 if S/n=0.388")
for x, S in runs.items():
    cost = Fmin*((x*x+2/x)/3-1)
    print(f"{x:4.1f}   {cost:6.3f}            {N*0.2*np.sqrt(S):6.3f}                       {N*0.05*np.sqrt(0.388):6.3f}")
