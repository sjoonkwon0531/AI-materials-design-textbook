# Ex 4.7.1: reference shift between 1/2 H2 and isolated H atom: Delta = E(H) - 1/2 E(H2) = D_e/2
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
# (a) EMT (the reference model of Lab 4.7)
h2 = molecule('H2'); h2.calc = EMT(); BFGS(h2, logfile=None).run(fmax=1e-4); eH2 = h2.get_potential_energy()
h = Atoms('H'); h.calc = EMT(); eH = h.get_potential_energy()
De_emt = 2*eH - eH2
print(f"EMT: E(H2)={eH2:.4f} eV (d={h2.get_distance(0,1):.3f} A), E(H)={eH:.4f} eV, D_e={De_emt:.3f} eV, D_e/2={De_emt/2:.3f} eV")
Eads_H2 = -0.5206                                      # Lab 4.7 fcc, 1/2 H2 reference
print(f"EMT: E_ads(fcc) vs 1/2 H2 = {Eads_H2:+.4f} eV -> vs H atom = {Eads_H2 - De_emt/2:+.4f} eV")
# (b) PBE molecular calculation (PySCF): spin-polarized H atom, H2 optimized by a 1D scan
from pyscf import gto, dft
def e_h2(d, basis='aug-cc-pvqz'):
    m = gto.M(atom=f'H 0 0 0; H 0 0 {d}', basis=basis, verbose=0); mf = dft.RKS(m); mf.xc='pbe'; return mf.kernel()
mH = gto.M(atom='H 0 0 0', basis='aug-cc-pvqz', spin=1, verbose=0); mf = dft.UKS(mH); mf.xc='pbe'; e_atom = mf.kernel()
from scipy.optimize import minimize_scalar
opt = minimize_scalar(e_h2, bounds=(0.70,0.80), method='bounded', options={'xatol':1e-4})
Ha = 27.211386245988
De_pbe = (2*e_atom - opt.fun)*Ha
print(f"PBE/aug-cc-pVQZ: d_eq={opt.x:.4f} A, D_e={De_pbe:.3f} eV, D_e/2={De_pbe/2:.3f} eV")
