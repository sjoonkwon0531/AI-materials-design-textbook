import numpy as np
exec(open('../../ch02/labs/b04_L1335.py').read())
xc,u0,u,mass=run_fvm()
print("initial mass = %.10f"%mass[0]); print("final   mass = %.10f"%mass[-1])
print("max relative drift over run = %.3e"%np.max(np.abs(mass-mass[0])/mass[0]))
print("steps",len(mass)-1, "max |u-u0|", np.abs(u-u0).max(), "min,max u",u.min(),u.max())
for cfl in [1.05]:
    try:
        _,_,ub,_=run_fvm(CFL=cfl); print("CFL",cfl,"max|u|",np.abs(ub).max())
    except Exception as e: print(e)
