"""Lab 61_polymorph -- composition-only descriptors cannot see structure.

Part A: Magpie (matminer ElementProperty, preset 'magpie') vectors of Cu in four
        crystal structures are identical, while the EMT energy per atom differs.
Part B: the matbench_expt_gap copy used in this chapter is keyed by composition
        (one row per reduced formula): polymorph information is absent by design.
Part C: cost of composition featurization (time per formula) on a subsample.
Runtime: < 1 min on 2 cores.
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from ase.build import bulk
from ase.calculators.emt import EMT
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

ep = ElementProperty.from_preset("magpie")

# ---------- Part A
def emt_energy(struct, a):
    if struct == "hcp":
        at = bulk("Cu", "hcp", a=a, c=a * np.sqrt(8 / 3))
    else:
        at = bulk("Cu", struct, a=a)
    at.calc = EMT()
    return at.get_potential_energy() / len(at)

# search ranges chosen around each structure's nearest-neighbour distance ~2.55 A
ranges = {"fcc": (3.3, 3.9), "hcp": (2.3, 2.8), "bcc": (2.6, 3.2), "sc": (2.2, 2.8)}
print("=== Part A: Cu polymorphs, EMT energy at optimized lattice constant ===")
E = {}
for s, (lo, hi) in ranges.items():
    r = minimize_scalar(lambda a: emt_energy(s, a), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-5})
    E[s] = r.fun
    print(f"  {s:4s} a = {r.x:.4f} A   E_EMT = {r.fun*1000:8.1f} meV/atom")
ref = E["fcc"]
print("  relative to fcc (meV/atom): " +
      ", ".join(f"{s} {1000*(E[s]-ref):+.1f}" for s in E))
v = np.array(ep.featurize(Composition("Cu")))
print(f"  Magpie vector length = {len(v)}; ||v(fcc)-v(bcc)|| = 0 by construction "
      f"(composition is the only input)")
spread = 1000 * (max(E.values()) - min(E.values()))
print(f"  any composition-only model must err by >= {spread/2:.1f} meV/atom on at least one of them")

# ---------- Part B
d = pd.read_csv("expt_gap_matbench.csv")
comp = [Composition(f) for f in d.formula]
red = [c.reduced_formula for c in comp]
nel = pd.Series([len(c.elements) for c in comp]).value_counts().sort_index()
print("=== Part B: expt_gap (matbench, CrabNet mirror) ===")
print(f"  rows = {len(d)}, unique reduced formulas = {len(set(red))}, "
      f"metals (gap = 0) = {(d.target == 0).sum()} ({100*(d.target == 0).mean():.1f}%)")
print("  number of elements per formula: " +
      ", ".join(f"{k}: {v}" for k, v in nel.items()))

# ---------- Part C
rng = np.random.default_rng(0)
idx = rng.choice(len(d), 300, replace=False)
t0 = time.perf_counter()
X = np.array([ep.featurize(comp[i]) for i in idx])
dt = time.perf_counter() - t0
print("=== Part C: Magpie featurization cost (300 random formulas, 1 process) ===")
print(f"  {dt/300*1e3:.2f} ms per formula; feature matrix {X.shape}; "
      f"NaN entries = {int(np.isnan(X).sum())}")
