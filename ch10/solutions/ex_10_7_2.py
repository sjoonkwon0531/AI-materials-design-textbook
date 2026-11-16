"""Exercise 10.7.2: tilt angle -> equatorial O displacement; Ti-O stretch of a rigid tilt; c/a of rigid octahedra."""
import numpy as np, json
a = 3.944
for phi in (4.9, 2.1):
    print(f"phi = {phi} deg: u = (a/2) tan(phi) = {a/2*np.tan(np.radians(phi)):.4f} A; Ti-O-Ti = {180-2*phi:.1f} deg;"
          f" rigid-tilt Ti-O stretch 1/cos(phi) - 1 = {100*(1/np.cos(np.radians(phi))-1):.3f} %")
print("model      phi_rel   1/cos(phi)   c/a(relaxed)   a_inplane/(2 a0)   c/(2 a0)")
for m in ('mace', 'chgnet', 'sevennet', 'mattersim'):
    d = json.load(open(f'../data/107_{m}.json')); t = d['tilt']; a0 = d['eos']['a0']; cp = t['cell']
    print(f"{m:10s} {t['phi_rel']:6.2f}   {1/np.cos(np.radians(t['phi_rel'])):.5f}      {cp[2]/cp[0]:.5f}        "
          f"{cp[0]/(2*a0):.5f}            {cp[2]/(2*a0):.5f}")
