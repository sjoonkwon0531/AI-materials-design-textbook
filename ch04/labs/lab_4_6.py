import numpy as np, warnings
warnings.filterwarnings("ignore")
from ase.build import bulk, fcc111, add_adsorbate, molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.eos import EquationOfState
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.constraints import FixAtoms
from ase.cluster.cubic import FaceCenteredCubic
np.random.seed(0)
# Lab 4.6  ASE workflows with EMT (recovered from v1.0 session).
print("--- (1) Calculator abstraction ---")
n2 = molecule('N2'); n2.calc = EMT()
print(f"  N2 potential energy = {n2.get_potential_energy():.4f} eV")
print(f"  N2 forces (eV/Ang):\n{np.round(n2.get_forces(),4)}")

print("\n--- (2) EOS: Cu fcc ---")
cu = bulk('Cu', 'fcc', a=3.6, cubic=True); cell0 = cu.get_cell()
vols, ens = [], []
for s in np.linspace(0.94, 1.06, 11):
    c = cu.copy(); c.set_cell(cell0*s, scale_atoms=True); c.calc = EMT()
    vols.append(c.get_volume()); ens.append(c.get_potential_energy())
eos = EquationOfState(vols, ens); v0, e0, B = eos.fit()
a_eq = (v0/len(cu)*4)**(1/3); B_GPa = B/units.kJ*1.0e24
print(f"  V0={v0:.3f} Ang^3, a={a_eq:.4f} Ang, B={B_GPa:.1f} GPa")

print("\n--- (3) BFGS: Cu cluster ---")
clus = FaceCenteredCubic('Cu', surfaces=[(1,0,0),(1,1,1)], layers=[1,1], latticeconstant=3.6)
clus.rattle(0.1, seed=1); clus.calc = EMT()
e_init = clus.get_potential_energy(); fmax_init = np.sqrt((clus.get_forces()**2).sum(1).max())
BFGS(clus, logfile=None).run(fmax=0.02, steps=200)
e_final = clus.get_potential_energy(); fmax_final = np.sqrt((clus.get_forces()**2).sum(1).max())
print(f"  atoms: {len(clus)} | E_init={e_init:.4f} (fmax={fmax_init:.3f}) -> E_final={e_final:.4f} (fmax={fmax_final:.4f}); dE={e_init-e_final:.4f}")

print("\n--- (4) Langevin NVT MD, Cu 4x4x4 ---")
sc = bulk('Cu','fcc',a=a_eq).repeat((4,4,4)); sc.calc = EMT()
MaxwellBoltzmannDistribution(sc, temperature_K=300, rng=np.random.RandomState(0))
dyn = Langevin(sc, timestep=2.0*units.fs, temperature_K=300, friction=0.02, fixcm=False, rng=np.random.RandomState(1))
Ts=[]; Es=[]
def rec():
    Ts.append(sc.get_temperature()); Es.append(sc.get_potential_energy()+sc.get_kinetic_energy())
dyn.attach(rec, interval=10); dyn.run(1500)
h=len(Ts)//2
print(f"  atoms={len(sc)} mean T(2nd half)={np.mean(Ts[h:]):.1f} +/- {np.std(Ts[h:]):.1f} K; E drift={Es[-1]-Es[0]:+.3f} eV")

print("\n--- (5) Pt(111) slab + H ---")
slab = fcc111('Pt', size=(2,2,4), vacuum=7.0)
slabH = slab.copy(); add_adsorbate(slabH, 'H', height=1.5, position='fcc'); slabH.calc = EMT()
zs = slab.positions[:,2]; cutz = np.sort(np.unique(np.round(zs,3)))[1]
slab.set_constraint(FixAtoms(mask=[z < cutz+0.1 for z in slab.positions[:,2]])); slab.calc = EMT()
BFGS(slab, logfile=None).run(fmax=0.05, steps=200); e_slab = slab.get_potential_energy()   # relaxed clean slab
slabH.set_constraint(FixAtoms(mask=[z < cutz+0.1 for z in slabH.positions[:,2]]))
BFGS(slabH, logfile=None).run(fmax=0.05, steps=200)
e_slabH = slabH.get_potential_energy()
h2 = molecule('H2'); h2.calc = EMT(); BFGS(h2, logfile=None).run(fmax=0.01); eH2 = h2.get_potential_energy()   # relaxed H2
print(f"  slab={e_slab:.4f} slab+H={e_slabH:.4f} H2={eH2:.4f}  E_ads={e_slabH-e_slab-0.5*eH2:.4f} eV")
print(f"  pbc={slab.pbc}, cell={np.round(slab.cell.lengths(),3)}")
np.savez('../ch04/labs/lab46_data.npz', vols=vols, ens=ens, v0=v0, a=a_eq, B=B_GPa, Ts=Ts)
