"""Exercise 10.6.1: residual pressure -> lattice-constant error, and its effect on the soft phonon."""
import numpy as np
B = 164.26                      # GPa, MACE-MP-0 small SrTiO3 (data/107_mace.json)
a = 3.94321                     # converged a (fmax 0.01 and 0.001)
for dP in (-0.1824, 0.0090):
    da = -a * dP / (3 * B)
    print(f"dP = {dP:+.4f} GPa -> da/a = {-dP/(3*B):+.2e}, da = {da:+.5f} A")
print(f"lab: a(fmax 0.05) - a(converged) = {3.94476-3.94321:+.5f} A")
# soft-mode sensitivity (MACE, Lab 10.12): G_TO1 106i at 3.944, 63 at 3.905; R4+ 71i at 3.944, 88i at 3.905
for name, w1, w2 in [('G_TO1', -106, 63), ('R4+', -71, -88)]:
    s = (np.sign(w2)*w2**2 - np.sign(w1)*w1**2) / (3.905 - 3.944)      # d(omega^2)/da, cm-2 per A
    for da in (0.00155, 0.0001):
        w2new = np.sign(w1)*w1**2 + s * da
        w = np.sign(w2new)*np.sqrt(abs(w2new))
        print(f"{name}: d(w^2)/da = {s:,.0f} cm-2/A; da = {da:.5f} A -> w = {abs(w):.1f}{'i' if w<0 else ''} cm-1 (from {abs(w1)}{'i' if w1<0 else ''})")
