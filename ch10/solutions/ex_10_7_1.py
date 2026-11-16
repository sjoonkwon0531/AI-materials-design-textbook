"""Exercise 10.7.1: number of displaced supercells phonopy needs for cubic and I4/mcm SrTiO3.
Run: /tmp/claude-0/v10/bin/python ex_10_7_1.py"""
import numpy as np, itertools
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import spglib

def sto(a, rep=1):
    at = Atoms('SrTiO3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
               cell=np.eye(3) * a, pbc=True)
    return at.repeat((rep, rep, rep)) if rep > 1 else at

def tilted(a, phi_deg, ca=1.0):          # same construction as labs/107_phonons.py, then a tetragonal strain
    at = sto(a, 2); u = 0.5 * a * np.tan(np.radians(phi_deg))
    for c in range(8):
        i, j, k = c // 4, (c // 2) % 2, c % 2
        s = (-1) ** (i + j + k)
        at.positions[5 * c + 4, 1] += -s * u
        at.positions[5 * c + 3, 0] += s * u
    cell = at.cell.array.copy(); cell[2] *= ca
    at.set_cell(cell, scale_atoms=True)
    return at

def pa(at):
    return PhonopyAtoms(symbols=at.get_chemical_symbols(), cell=at.cell[:], scaled_positions=at.get_scaled_positions())

u = sto(3.944)
ph = Phonopy(pa(u), supercell_matrix=np.eye(3, dtype=int) * 2)
ph.generate_displacements(distance=0.01)
print("cubic Pm-3m, 2x2x2 (40 atoms):", len(ph.supercells_with_displacements), "displaced supercells")
syms = ph.supercell.symbols
for d in ph.displacements:
    print(f"   atom {d[0]:2d} ({syms[d[0]]}): displacement {np.round(np.array(d[1:]) / 0.01, 3)} x 0.01 A")
# O site symmetry: images of the diagonal displacement under the O site-symmetry operations span 3-D?
ds = spglib.get_symmetry_dataset((u.cell[:], u.get_scaled_positions(), u.numbers))
print("site symmetries (Sr, Ti, O, O, O):", list(ds.site_symmetry_symbols))
rots = [r for r, t in zip(ds.rotations, ds.translations)
        if np.allclose((r @ u.get_scaled_positions()[2] + t - u.get_scaled_positions()[2] + 0.5) % 1 - 0.5, 0)]
v = np.array([1, 0, 1]) / np.sqrt(2)
img = np.array([r @ v for r in rots])
print(f"O at (1/2,1/2,0): {len(rots)} site-symmetry operations; rank of the images of (1,0,1)/sqrt2 = {np.linalg.matrix_rank(img)}")
img_x = np.array([r @ np.array([1., 0, 0]) for r in rots]); img_z = np.array([r @ np.array([0, 0, 1.]) for r in rots])
print(f"   rank of images of (1,0,0) = {np.linalg.matrix_rank(img_x)}, of (0,0,1) = {np.linalg.matrix_rank(img_z)}")
# no symmetry at all
ph0 = Phonopy(pa(u), supercell_matrix=np.eye(3, dtype=int) * 2, is_symmetry=False)
ph0.generate_displacements(distance=0.01, is_plusminus=False)
print("is_symmetry=False (lattice translations of the supercell still used by phonopy):", len(ph0.supercells_with_displacements))
print("fully naive count, every atom of the 40-atom supercell x 3 directions:", 40 * 3, "(x2 with +/-:", 240, ")")
# tetragonal I4/mcm: tilted 2x2x2 cell (40 atoms), relaxed-like tilt 4.75 deg and c/a 1.004
t = tilted(3.9433, 4.75, 1.004)
print("tilted cell space group:", spglib.get_spacegroup((t.cell[:], t.get_scaled_positions(), t.numbers), symprec=1e-4))
for sc_label, unit, scm in [("I4/mcm, the 40-atom tilted cell as the supercell", t, np.eye(3, dtype=int)),
                            ("I4/mcm, 2x2x2 of the 40-atom cell (320 atoms)", t, np.eye(3, dtype=int) * 2)]:
    p = Phonopy(pa(unit), supercell_matrix=scm, primitive_matrix='auto', symprec=1e-4)
    p.generate_displacements(distance=0.01)
    print(f"{sc_label}: primitive cell {len(p.primitive)} atoms, {len(p.supercells_with_displacements)} displaced supercells")
    print("   displaced atoms:", [p.supercell.symbols[d[0]] for d in p.displacements])
dt = spglib.get_symmetry_dataset((t.cell[:], t.get_scaled_positions(), t.numbers), symprec=1e-4)
seen = {}
for s, w, ss in zip(t.get_chemical_symbols(), dt.wyckoffs, dt.site_symmetry_symbols):
    seen.setdefault((s, w), ss)
print("I4/mcm Wyckoff positions and site symmetries:", seen)
