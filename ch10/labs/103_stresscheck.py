"""Diagnostic for Lab 10.3: is the model's stress the strain derivative of its own energy?
Si diamond (8-atom cubic cell) at the model's EOS a0; compare (i) stress from the calculator with
(ii) the central finite difference of E under a symmetric strain, and the bulk modulus from
(a) E(V) curvature at +-0.5 % and +-1 % strain, (b) dP/dlnV from calculator stress."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('103_common'); model = sys.argv[1]; C.reexec_for(model)
import numpy as np
from ase.build import bulk
from ase import units
calc, net, info = C.make_calc(model)
d = json.load(open(os.path.join(C.DATA, f'103_{model}.json')))['eos']
out = {}
for sym, cr, nat in (('Si', 'diamond', 8), ('Cu', 'fcc', 4)):
    a0 = (d[sym]['V0'] * nat) ** (1 / 3); base = bulk(sym, cr, a=a0, cubic=True)
    def at(eps_iso=0.0, e=None):
        a = base.copy(); F = np.eye(3) * (1 + eps_iso) if e is None else np.eye(3) + e
        a.set_cell(base.cell @ F, scale_atoms=True); a.calc = calc; return a
    V = base.get_volume(); h = 1e-3
    # (i) stress_xx vs finite difference of E in e_xx
    s_calc = at(0.0).get_stress(voigt=True)[0]
    Ep = at(e=np.diag([h, 0, 0])).get_potential_energy(); Em = at(e=np.diag([-h, 0, 0])).get_potential_energy()
    s_fd = (Ep - Em) / (2 * h) / V
    # B from energy curvature and from stress
    res = {}
    for x in (0.005, 0.01):
        E = [at(s).get_potential_energy() for s in (-x, 0, x)]
        # V(eps) = V(1+eps)^3 ; B = V d2E/dV2 ; use E(eps) curvature: d2E/deps2 = 9 V B at eps=0 (to O(P))
        res[f'B_E_{x}'] = float((E[0] - 2 * E[1] + E[2]) / x ** 2 / (9 * V) / units.GPa)
        P = [-np.trace(at(s).get_stress(voigt=False)) / 3 for s in (-x, x)]
        res[f'B_S_{x}'] = float(-(P[1] - P[0]) / (3 * 2 * x) / units.GPa)
    res.update(stress_xx_calc_GPa=float(s_calc / units.GPa), stress_xx_fd_GPa=float(s_fd / units.GPa))
    out[sym] = res
    print(f"[{model}] {sym}: sigma_xx calc = {res['stress_xx_calc_GPa']:+.3f} GPa, FD of E = {res['stress_xx_fd_GPa']:+.3f} GPa; "
          f"B from E'' (0.5%/1%) = {res['B_E_0.005']:.1f}/{res['B_E_0.01']:.1f}, from dP (0.5%/1%) = {res['B_S_0.005']:.1f}/{res['B_S_0.01']:.1f} GPa")
C.save(model, 'stresscheck', out)
