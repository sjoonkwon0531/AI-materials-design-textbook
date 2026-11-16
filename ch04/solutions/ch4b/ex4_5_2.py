# Ex 4.5.2: atomwise decomposition -> extensivity / permutation invariance / size transfer (EMT is itself E=sum_i eps_i)
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase import Atoms
from ase.build import bulk
from ase.cluster.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
A = Atoms(FaceCenteredCubic('Cu', surfaces=[(1,0,0),(1,1,1)], layers=[1,1], latticeconstant=3.6))
def E(a): a=a.copy(); a.calc=EMT(); return a.get_potential_energy(), a.get_potential_energies()
EA, epsA = E(A)
print(f"(i) E(A)={EA:.6f} eV, sum_i eps_i = {epsA.sum():.6f} eV")
B = A.copy(); B.translate([30,0,0]); AB = A + B
EAB,_ = E(AB); print(f"(a) E(A+B, 30 A apart) - 2E(A) = {EAB-2*EA:.2e} eV")
# (c) per-atom energy of bulk is size independent: 1x1x1 ... 4x4x4 supercells
for n in [1,2,3,4]:
    s = bulk('Cu','fcc',a=3.59,cubic=True).repeat((n,n,n)); e,_ = E(s)
    print(f"(c) Cu bulk {len(s):4d} atoms: E/N = {e/len(s):.6f} eV/atom")
