"""Exercise 10.7.3: Landau double well E = c2 u^2 + c4 u^4 (per f.u.) from the Lab 10.13 fits."""
import numpy as np, json
fits = {'mace': (-0.293, 6.16), 'chgnet': (-0.129, 4.40), 'sevennet': (-0.254, 5.42), 'mattersim': (-0.222, 5.97)}
rigid = {'mace': (-3.46, 4.5), 'chgnet': (-0.96, 4.0), 'sevennet': (-2.98, 4.5), 'mattersim': (-2.06, 4.0)}
print("model      u* (A)   phi* (deg)  dE (meV/f.u.) | rigid-tilt min | refit c2,c4 from the stored profile")
for m, (c2, c4) in fits.items():
    d = json.load(open(f'../data/107_{m}.json')); a0 = d['eos']['a0']; t = d['tilt']
    us = np.sqrt(-c2 / (2 * c4)); phi = np.degrees(np.arctan(2 * us / a0)); dE = -c2 ** 2 / (4 * c4) * 1000
    # refit with full precision from the stored rigid-tilt profile (same recipe as labs/107_compare.py)
    ph, E = np.array(t['phis']), np.array(t['prof']) / 1000; u = a0 / 2 * np.tan(np.radians(ph)); s = ph <= 6
    C2, C4 = np.linalg.lstsq(np.vstack([u[s] ** 2, u[s] ** 4]).T, E[s], rcond=None)[0]
    us2 = np.sqrt(-C2 / (2 * C4)); dE2 = -C2 ** 2 / (4 * C4) * 1000
    print(f"{m:10s} {us:.4f}   {phi:5.2f}       {dE:6.2f}       | {rigid[m][0]:6.2f} at {rigid[m][1]} | "
          f"c2 {C2:.4f}, c4 {C4:.3f}: u* {us2:.4f}, phi* {np.degrees(np.arctan(2*us2/a0)):.2f}, dE {dE2:.2f}; "
          f"relaxed I4/mcm {t['dE_rel']:.2f} meV at {t['phi_rel']:.2f} deg")
print("depth scaling with s (c4 fixed): dE ~ c2^2 -> s^2; e.g. s = 0.6 -> depth x", 0.6 ** 2)
