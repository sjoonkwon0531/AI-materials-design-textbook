# Ex 4.6.3: EMT H adsorption energy on Pt(111) at fcc/hcp/top/bridge (Lab 4.6A(5) settings)
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ase.build import fcc111, add_adsorbate, molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FixAtoms
def relax(a, fmax=0.05):
    a.calc = EMT(); BFGS(a, logfile=None).run(fmax=fmax, steps=3000); return a.get_potential_energy()
h2 = molecule('H2'); eH2 = relax(h2, 0.01)
def fixed(a, cut): a.set_constraint(FixAtoms(mask=a.positions[:,2] < cut)); return a
slab0 = fcc111('Pt', size=(2,2,4), vacuum=7.0)
cut = np.sort(np.unique(np.round(slab0.positions[:,2],3)))[1] + 0.1
E_clean = relax(fixed(slab0.copy(), cut))
print(f"E(H2)={eH2:.4f} eV, E(clean slab, relaxed)={E_clean:.4f} eV")
print(" site     E_ads(eV)  H height(A)  shortest Pt-H (A)  #Pt within 2.1 A   lateral shift (A)")
res = {}
for site in ['fcc','hcp','bridge','ontop']:
    s = slab0.copy(); add_adsorbate(s, 'H', height=1.5, position=site); s = fixed(s, cut)
    xy0 = s.positions[-1,:2].copy(); E = relax(s)
    ztop = s.positions[:-1,2].max(); d = s.get_distances(len(s)-1, range(len(s)-1), mic=True)
    res[site] = E - E_clean - 0.5*eH2
    print(f" {site:7s} {res[site]:+.4f}    {s.positions[-1,2]-ztop:.3f}        {d.min():.3f}             {np.sum(d<2.1)}            {np.linalg.norm(s.positions[-1,:2]-xy0):.3f}")
best = min(res, key=res.get); print(f"most stable (EMT): {best}; E_ads(fcc)-E_ads(top) = {res['fcc']-res['ontop']:+.4f} eV")
# are bridge/top true minima? break the symmetry by a 0.15 A in-plane kick and re-relax (fmax=0.01)
print("symmetry-breaking test (0.15 A lateral kick, fmax=0.001):")
for site in ['bridge','ontop']:
    s = slab0.copy(); add_adsorbate(s, 'H', height=1.5, position=site); s = fixed(s, cut)
    xy0 = s.positions[-1,:2].copy(); s.positions[-1,:2] += [0.15*np.cos(0.3), 0.15*np.sin(0.3)]
    E = relax(s, 0.001) - E_clean - 0.5*eH2
    print(f"  start {site:7s}: E_ads = {E:+.4f} eV, final lateral displacement from start site = {np.linalg.norm(s.positions[-1,:2]-xy0):.3f} A")
