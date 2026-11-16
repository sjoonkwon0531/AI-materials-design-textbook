# Ex 4.5.1: Cartesian coordinates are not invariant, energy is (EMT Cu13 cluster)
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase.cluster.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
from scipy.spatial.transform import Rotation
at = FaceCenteredCubic('Cu', surfaces=[(1,0,0),(1,1,1)], layers=[1,1], latticeconstant=3.6)
from ase import Atoms
at = Atoms(at); at.rattle(0.1, seed=1)
def E(a): a=a.copy(); a.calc=EMT(); return a.get_potential_energy()
x0 = at.positions.ravel().copy(); E0 = E(at)
tr = at.copy(); tr.translate([1.0,0,0])
ro = at.copy(); R = Rotation.from_euler('z', 30, degrees=True).as_matrix()
ro.positions = at.positions @ R.T
pe = at.copy(); idx = np.arange(len(at)); idx[[0,5]] = idx[[5,0]]; pe = pe[idx]
print(f"N atoms = {len(at)}, E0 = {E0:.6f} eV")
for name,a in [("translate 1 A",tr),("rotate 30 deg",ro),("swap atoms 0,5",pe)]:
    dx = np.linalg.norm(a.positions.ravel()-x0)
    print(f"{name:15s}: ||x'-x|| = {dx:7.3f} A,  |E'-E0| = {abs(E(a)-E0):.1e} eV")
# distance-matrix descriptor (sorted) is invariant
def desc(a):
    d = a.get_all_distances(); return np.sort(d[np.triu_indices(len(a),1)])
for name,a in [("translate",tr),("rotate",ro),("permute",pe)]:
    print(f"sorted-distance descriptor change ({name}): {np.abs(desc(a)-desc(at)).max():.1e} A")
