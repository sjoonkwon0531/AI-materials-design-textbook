"""Exercise 10.6.4: break-even cost ratio rho for ML pre-relaxation, and a 1-D double-well model in which
pre-relaxation increases the number of reference calls."""
import numpy as np
from scipy.optimize import minimize
for name, nraw, nml, npre in [('SrTiO3 rattled', 21, 19, 5), ('Si vacancy', 15, 22, 17)]:
    rho = (nraw - npre) / nml
    print(f"{name}: (n_raw - n_pre)/n_ML = ({nraw}-{npre})/{nml} = {rho:.3f}" + ("  -> never profitable" if rho <= 0 else ""))

# 1-D model.  Reference: E_ref = (x^2-1)^2 + 0.3 x  -> minima near -1.04 (global) and +0.96 (local).
# uMLIP: a shifted copy whose minima are displaced by dx (a systematically wrong structure).
def make(shift):
    E = lambda x: ((x[0] - shift) ** 2 - 1) ** 2 + 0.3 * (x[0] - shift)
    G = lambda x: np.array([4 * (x[0] - shift) * ((x[0] - shift) ** 2 - 1) + 0.3])
    return E, G
Eref, Gref = make(0.0)
def relax(E, G, x0, gtol=1e-3, eta=0.05):
    # fixed-step steepest descent: the call count grows with the distance to walk (BFGS in 1-D hides this)
    x = np.array([x0]); n = 1
    g = G(x)
    while abs(g[0]) > gtol and n < 100000:
        x = x - eta * g; g = G(x); n += 1
    return x[0], n
x0 = -0.75
xr, nraw = relax(Eref, Gref, x0)
print(f"reference only from x0 = {x0}: x = {xr:+.4f}, reference gradient calls {nraw}")
for shift in (0.05, -0.3, -0.6, 0.5, 0.8, 1.2):
    Eml, Gml = make(shift)
    xm, nml = relax(Eml, Gml, x0)
    xp, npre = relax(Eref, Gref, xm)
    print(f"uMLIP minima shifted by {shift:+.2f}: ML relax -> {xm:+.4f} ({nml} ML calls), then reference -> {xp:+.4f} "
          f"({npre} reference calls; raw {nraw}){'  DIFFERENT MINIMUM' if abs(xp - xr) > 1e-2 else ''}")
