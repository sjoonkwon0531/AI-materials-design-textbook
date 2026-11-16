# Ex 4.6.1: EOS of fcc Al with EMT; sensitivity to the strain window
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
from ase import units
al = bulk('Al','fcc',a=4.05,cubic=True); cell0 = al.get_cell()
def eos(lo, hi, n=11, kind='sj'):
    V,E=[],[]
    for s in np.linspace(lo,hi,n):
        c=al.copy(); c.set_cell(cell0*s, scale_atoms=True); c.calc=EMT()
        V.append(c.get_volume()); E.append(c.get_potential_energy())
    v0,e0,B = EquationOfState(V,E,eos=kind).fit()
    return (v0)**(1/3), B/units.kJ*1e24          # cubic 4-atom cell: a = V0^(1/3)
print("window        a0 (sj)   B (sj)   |  a0 (BM)   B (BM)")
for lo,hi in [(0.99,1.01),(0.97,1.03),(0.94,1.06),(0.90,1.10),(0.85,1.15)]:
    a1,B1 = eos(lo,hi); a2,B2 = eos(lo,hi,kind='birchmurnaghan')
    print(f"{lo:.2f}-{hi:.2f}   {a1:.4f}  {B1:7.1f}   |  {a2:.4f}  {B2:7.1f}")
# diagnose: local curvature d2E/da2 (eV/atom/A^2) along the 0.94-1.06 window -> EMT cutoff-shell wiggles
ss=np.linspace(0.94,1.06,49); Ea=[]
for s in ss:
    c=al.copy(); c.set_cell(cell0*s, scale_atoms=True); c.calc=EMT(); Ea.append(c.get_potential_energy()/len(c))
a=4.05*ss; d2=np.diff(Ea,2)/(a[1]-a[0])**2
print(f"d2E/da2 on 0.94-1.06: min={d2.min():.2f} at a={a[1:-1][d2.argmin()]:.3f}, max={d2.max():.2f} at a={a[1:-1][d2.argmax()]:.3f} (eV/atom/A^2)")
