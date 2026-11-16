# shared helpers for Pt(111)+H EMT exercises (same protocol as Lab 4.7)
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase.build import fcc111, add_adsorbate, molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FixAtoms
def relax(a, fmax=0.05):
    a.calc = EMT(); BFGS(a, logfile=None).run(fmax=fmax, steps=500); return a.get_potential_energy()
_h2 = molecule('H2'); eH2 = relax(_h2, 0.01)
def E_ads_site(site, size=(2,2,4), vacuum=8.0, h=1.4, return_atoms=False):
    slab = fcc111('Pt', size=size, vacuum=vacuum)
    layers = np.sort(np.unique(np.round(slab.positions[:,2],2))); cut = layers[1] + 0.1
    base = slab.copy(); base.set_constraint(FixAtoms(mask=[z < cut for z in base.positions[:,2]]))
    E_clean = relax(base)
    sl = slab.copy(); add_adsorbate(sl, 'H', height=h, position=site)
    sl.set_constraint(FixAtoms(mask=[z < cut for z in sl.positions[:,2]]))
    E = relax(sl) - E_clean - 0.5*eH2
    return (E, sl) if return_atoms else E
