import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase.build import fcc111, add_adsorbate, molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FixAtoms
# Lab 4.7  H adsorption on Pt(111) with EMT (recovered from v1.0 session).
def relax(atoms, fmax=0.05):
    atoms.calc = EMT(); BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
    return atoms.get_potential_energy()
h2 = molecule('H2'); h2.calc = EMT(); BFGS(h2, logfile=None).run(fmax=0.01); eH2 = h2.get_potential_energy()   # relaxed H2 reference
def E_ads_site(site, size=(2,2,4), vacuum=8.0, h=1.4):
    slab = fcc111('Pt', size=size, vacuum=vacuum)
    layers = np.sort(np.unique(np.round(slab.positions[:,2],2))); cut = layers[1] + 0.1
    base = slab.copy(); base.set_constraint(FixAtoms(mask=[z < cut for z in base.positions[:,2]]))
    E_clean = relax(base)
    sl = slab.copy(); add_adsorbate(sl, 'H', height=h, position=site)
    sl.set_constraint(FixAtoms(mask=[z < cut for z in sl.positions[:,2]]))
    return relax(sl) - E_clean - 0.5*eH2
print(f"H2 reference (EMT) = {eH2:.4f} eV")
sites=['fcc','hcp','bridge','ontop']; Eads={s:E_ads_site(s) for s in sites}
for s in sites: print(f"  {s:8s} {Eads[s]:+.4f}")
lay=[3,4,5,6]; Elay=[E_ads_site('fcc', size=(2,2,n)) for n in lay]
for n,e in zip(lay,Elay): print(f"  layers {n}: {e:+.4f}")
vac=[5,7,9,12]; Evac=[E_ads_site('fcc', vacuum=float(v)) for v in vac]
for v,e in zip(vac,Evac): print(f"  vacuum {v}: {e:+.4f}")
np.savez('../../ch04/labs/lab47_data.npz', sites=sites, Eads=[Eads[s] for s in sites], lay=lay, Elay=Elay, vac=vac, Evac=Evac)
