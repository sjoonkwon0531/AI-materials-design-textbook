# Ex 2.6.6: off-critical quench <c>=0.4 -> droplets of the minority phase; compare with symmetric <c>=0
import numpy as np
from scipy import ndimage
from ch_common import ch_run, length_scale

def periodic_components(mask):
    lab, n = ndimage.label(mask)
    parent = list(range(n+1))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in list(zip(lab[0, :], lab[-1, :])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b: parent[find(a)] = find(b)
    return len({find(i) for i in range(1, n+1)})

def spans(mask):
    """does a component wrap/connect the two opposite edges (x direction)? (non-periodic labelling)"""
    lab, n = ndimage.label(mask)
    return bool(set(lab[0, :][lab[0, :] > 0]) & set(lab[-1, :][lab[-1, :] > 0]))

for mean in [0.0, 0.4]:
    snaps = {}
    def cb(t, c, ch):
        for ts in (5, 20, 100, 400):
            if abs(t-ts) < 1e-9: snaps[ts] = c.copy()
    ch_run(mean=mean, T=400.0, seed=1, callback=cb, every=20)
    print(f"\n<c>0 = {mean}:  (minority = c<0 for <c>=0.4)")
    for ts in (5, 20, 100, 400):
        c = snaps[ts]; minor = c < 0
        print(f"  t={ts:4d}: mean={c.mean():+.4f}  area(c<0)={minor.mean():.3f}  #components(c<0)={periodic_components(minor):4d}"
              f"  #components(c>0)={periodic_components(~minor):3d}  c<0 spans box: {spans(minor)}   L(t)={length_scale(c):.3f}")
print("\nlever rule for <c>=0.4 with phases at -1,+1: minority fraction = (1-0.4)/2 = 0.30")
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
    for a, m in zip(ax, [0.0, 0.4]):
        a.imshow(ch_run(mean=m, T=100.0, seed=1), cmap='RdBu_r', vmin=-1, vmax=1); a.set_axis_off(); a.set_title(f'<c>={m}, t=100')
    fig.tight_layout(); fig.savefig('ex2_6_6_morphology.png', dpi=110); print("saved ex2_6_6_morphology.png")
except Exception as ex:
    print("figure skipped:", ex)
