"""Ch.3 exercise solutions: convex-hull exercises 3.5.C, 3.6.A, 3.6.B (pymatgen, local).
Formation energies are the book's Lab 3.3 teaching values (eV/atom), not DB values.
"""
from pymatgen.core import Composition, Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen.core, importlib.metadata as md
print("pymatgen", md.version("pymatgen"))

def build(extra=None, li2o=-2.05):
    data = [("Li", 0.0), ("O", 0.0), ("Li2O", li2o), ("Li2O2", -1.58), ("LiO2", -0.90)]
    if extra: data += extra
    return [PDEntry(Composition(f), e * Composition(f).num_atoms) for f, e in data]

def report(entries, title):
    pd = PhaseDiagram(entries)
    print(f"--- {title}")
    for e in entries:
        xo = e.composition.get_atomic_fraction("O")
        ef, eh = pd.get_form_energy_per_atom(e), pd.get_e_above_hull(e)
        tag = "STABLE" if eh < 1e-8 else f"{eh:+.3f} eV/atom above hull"
        print(f"  {e.composition.reduced_formula:<6s} x_O={xo:.3f} E_form={ef:+.3f} -> {tag}")
    print("  stable:", sorted(x.composition.reduced_formula for x in pd.stable_entries))
    return pd

def mu_ranges(pd, el="O"):
    # range of mu_O (relative to O2 reference, eV) over which each stable phase is stable
    from pymatgen.core import Element
    tr = sorted(pd.get_transition_chempots(Element(el)))
    print(f"  mu_{el} transition values (eV, ref=elemental):", [round(t, 3) for t in tr])
    return tr

print("\n=== Lab 3.3 baseline ===")
pd0 = report(build(), "baseline")
mu_ranges(pd0)

print("\n=== 3.5.C: Li2O 0.20 eV/atom lower (mixed functional) ===")
pd1 = report(build(li2o=-2.25), "Li2O = -2.25 eV/atom")
mu_ranges(pd1)
# threshold at which Li2O2 leaves the hull: line Li2O(1/3)->O2(1) at x=0.5 equals -1.58
# E(0.5) = E_Li2O * (1-0.5)/(1-1/3) = 0.75*E_Li2O
print(f"  Li2O2 destabilised when E_Li2O < {-1.58/0.75:.4f} eV/atom")
# also the opposite sign: Li2O 0.20 higher
pd1b = report(build(li2o=-1.85), "Li2O = -1.85 eV/atom (0.20 higher, for contrast)")

print("\n=== 3.6.B: add Li3O at -1.7 eV/atom ===")
pd2 = report(build(extra=[("Li3O", -1.7)]), "with Li3O")
mu_ranges(pd2)
# hull value at x_O=0.25 before adding Li3O (Li - Li2O tie line)
print(f"  hull at x_O=0.25 without Li3O: {-2.05*0.25/(1/3):.4f} eV/atom")
# threshold Li3O must beat
for e3 in (-1.5, -1.7):
    p = PhaseDiagram(build(extra=[("Li3O", e3)]))
    li3o = [x for x in p.all_entries if x.composition.reduced_formula == "Li3O"][0]
    print(f"  Li3O at {e3}: e_above_hull = {p.get_e_above_hull(li3o):+.4f}, decomposition energy = {p.get_equilibrium_reaction_energy(li3o) if p.get_e_above_hull(li3o)<1e-8 else 'n/a'}")

print("\n=== 3.6.A: Si density vs a ===")
for a in (5.43, 5.50):
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(a), ["Si"], [[0, 0, 0]])
    print(f"  a={a:.2f}  V={si.volume:.2f} A^3  density={si.density:.4f} g/cm^3")
r = (5.43/5.50)**3
print(f"  (5.43/5.50)^3 = {r:.4f}; 2.3302*ratio check; relative change = {(r-1)*100:.2f} %")
