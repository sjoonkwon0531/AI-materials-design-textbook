"""Ex 5.4.2: hexadecane, all-atom vs MARTINI cost estimate."""
heavy = 16; beads = heavy // 4
nmol = 10_000
aa_part = 50 * nmol; aa_steps = 1e-6 / 2e-15
cg_part = beads * nmol; cg_steps = (1e-6 / 4) / 20e-15
print(f"beads/molecule = {beads}")
print(f"all-atom: particles = {aa_part:.1e}, steps = {aa_steps:.1e}, particle-steps = {aa_part*aa_steps:.2e}")
print(f"MARTINI : particles = {cg_part:.1e}, steps = {cg_steps:.2e}, particle-steps = {cg_part*cg_steps:.2e}")
print(f"ratio = {aa_part*aa_steps/(cg_part*cg_steps):.0f}  (= particles {aa_part/cg_part:.1f} x dt {20/2:.0f} x time factor 4)")
