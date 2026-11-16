"""Fig 3.4 - forward (many-to-one) vs inverse (one-to-many) on a property landscape."""
import sys
sys.path.insert(0, '../../ch02/figure_scripts')
import bookstyle as bs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

def f(x1, x2):
    return (2.8*np.exp(-((x1-0.55)**2/0.18 + x2**2/0.35))
            + 0.6*np.exp(-((x1-0.45)**2/0.05 + (x2-1.0)**2/0.05)) * (-1.0)
            + 0.4*x1)

n = 400
x1, x2 = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
F = f(x1, x2)
ystar = 1.6

cmap = LinearSegmentedColormap.from_list('ice', ['#FFFFFF', bs.ICE, '#7FA3D9', bs.BLUE, bs.NAVY])
fig, ax = plt.subplots(figsize=(10, 8.2))
cf = ax.contourf(x1, x2, F, levels=14, cmap=cmap)
ax.contour(x1, x2, F, levels=14, colors='white', linewidths=0.6, alpha=0.7)
cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('property  $f(x)$')

# invalid / unstable region (hatched, lower-left wedge)
poly = Polygon([[0, 0], [0.42, 0], [0, 0.56]], closed=True, fc=bs.RED, alpha=0.30,
               ec=bs.RED, lw=2, hatch='xx', zorder=3)
ax.add_patch(poly)
ax.text(0.03, 0.085, 'invalid /\nunstable', color=bs.RED, fontsize=19, fontweight='bold',
        zorder=6, linespacing=1.0, bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))

# target level set f = y*
cs = ax.contour(x1, x2, F, levels=[ystar], colors=[bs.GOLD], linewidths=5, zorder=4)
path = max(cs.allsegs[0], key=len)
# dashed overlay where the level set lies inside the invalid region
inside = path[:, 1] < 0.56 - path[:, 0]*0.56/0.42 + 0.02
ax.plot(path[inside, 0], path[inside, 1], color=bs.RED, lw=2.2, ls=(0, (2, 2)), zorder=5)

# pick 5 valid solutions along the level set, well inside the axes
valid = path[(~inside) & (path[:, 0] < 0.93) & (path[:, 0] > 0.2) & (path[:, 1] > 0.05)]
s = np.r_[0, np.cumsum(np.hypot(*np.diff(valid, axis=0).T))]
picks = [valid[np.argmin(abs(s - q))] for q in np.linspace(0.04, 0.96, 5) * s[-1]]
picks = sorted(picks, key=lambda p: -p[0])
OFF = {1: (-0.035, 0.0, 'right', 'center'), 2: (0.03, 0.03, 'left', 'bottom'),
       3: (0.0, 0.035, 'center', 'bottom'), 4: (-0.02, 0.035, 'right', 'bottom'),
       5: (0.035, 0.0, 'left', 'center')}
for i, (px, py) in enumerate(picks, 1):
    ax.plot(px, py, 'o', ms=15, mfc=bs.NAVY, mec='white', mew=2.2, zorder=7)
    dx, dy, ha, va = OFF[i]
    ax.text(px + dx, py + dy, f'sol {i}', ha=ha, va=va, fontsize=18,
            fontweight='bold', color=bs.NAVY, zorder=7,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))

# direct label of the level set
ax.text(0.60, 0.66, r'target level set  $f = y^{*}$', fontsize=19, color='#8A5A00',
        fontweight='bold', ha='center', zorder=7,
        bbox=dict(fc='white', ec=bs.GOLD, lw=1.5, pad=3))

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('structure descriptor  $x_1$')
ax.set_ylabel('structure descriptor  $x_2$')
ax.spines[['top', 'right']].set_visible(True)
fig.savefig('../../ch03/figures/Fig3.4_forward_inverse.png')
