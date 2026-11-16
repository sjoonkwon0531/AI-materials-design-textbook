# Ex 4.6.2: one Atoms object, attach/detach calculators, BFGS bond length of Cu2
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
cu2 = Atoms('Cu2', positions=[[0,0,0],[0,0,2.8]])
print("calc before attach:", cu2.calc)
cu2.calc = EMT(); e0 = cu2.get_potential_energy()
BFGS(cu2, logfile=None).run(fmax=1e-4)
print(f"EMT : E(2.80 A) = {e0:.4f} eV -> d_eq = {cu2.get_distance(0,1):.4f} A, E = {cu2.get_potential_energy():.4f} eV")
cu2.calc = None                                   # detach: the structure survives, only the energy source is gone
print("calc after detach:", cu2.calc, "| positions kept, d =", round(cu2.get_distance(0,1),4))
cu2.calc = LennardJones(sigma=2.33, epsilon=0.40, rc=6.0)   # swap source of forces (illustrative parameters)
BFGS(cu2, logfile=None).run(fmax=1e-4)
print(f"LJ  : d_eq = {cu2.get_distance(0,1):.4f} A (analytic 2^(1/6)*sigma = {2**(1/6)*2.33:.4f} A)")
