"""Fig 3.2 - design -> make -> test discovery cycle, closed by data (MII hub)."""
import sys
sys.path.insert(0, '../../ch02/figure_scripts')
import bookstyle as bs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 9.2))
ax.set_xlim(-6.6, 6.6); ax.set_ylim(-5.3, 5.3); ax.set_aspect('equal'); ax.axis('off')

R = 3.6          # ring radius
r = 1.45         # node radius
ang = {'D': 90, 'M': -30, 'T': 210}
pos = {k: np.array([R*np.cos(np.radians(a)), R*np.sin(np.radians(a))]) for k, a in ang.items()}

nodes = {'D': ('DESIGN', 'predict / compute', bs.NAVY, 'white'),
         'M': ('MAKE', 'synthesis', bs.BLUE, 'white'),
         'T': ('TEST', 'characterize', bs.PURPLE, 'white')}

# arcs along the ring from node to node, with arrowheads
def ring_arc(a0, a1, color, lw, ls='-'):
    # a0 -> a1 (clockwise if a1 < a0); trim by node angular radius
    d = np.degrees(2*np.arcsin(r/(2*R))) + 2.0
    step = -1 if a1 < a0 else 1
    t = np.radians(np.linspace(a0 + step*d, a1 - step*d, 60))
    xy = np.c_[R*np.cos(t), R*np.sin(t)]
    ax.plot(xy[:-2, 0], xy[:-2, 1], color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=2)
    ax.add_patch(FancyArrowPatch(xy[-6], xy[-1], arrowstyle='-|>', mutation_scale=38,
                                 color=color, lw=lw, ls='-', shrinkA=0, shrinkB=0, zorder=3))
    return t

# Design -> Make (accelerated, gold), Make -> Test (bottleneck, red, dashed-free thick),
# Test -> Design (data feedback, gray dashed)
ring_arc(90, -30, bs.GOLD, 6)
ring_arc(-30, -150, bs.RED, 6, ls=(0, (1.2, 0.8)))
ring_arc(210, 90, bs.GRAY, 4, ls=(0, (4, 2)))

for k, (t1, t2, fc, tc) in nodes.items():
    ax.add_patch(Circle(pos[k], r, fc=fc, ec='white', lw=2, zorder=4))
    ax.text(*pos[k] + [0, 0.25], t1, ha='center', va='center', color=tc,
            fontsize=24, fontweight='bold', zorder=5)
    ax.text(*pos[k] - [0, 0.38], t2, ha='center', va='center', color=tc, fontsize=17, zorder=5)

# central data hub with spokes (data in, data out)
ax.add_patch(Circle((0, 0), 1.25, fc=bs.ICE, ec=bs.NAVY, lw=2, zorder=4))
ax.text(0, 0.18, 'DATA', ha='center', va='center', fontsize=23, fontweight='bold', color=bs.NAVY, zorder=5)
ax.text(0, -0.36, '& MII', ha='center', va='center', fontsize=19, fontweight='bold', color=bs.NAVY, zorder=5)
for k in 'DMT':
    u = pos[k] / np.linalg.norm(pos[k])
    ax.add_patch(FancyArrowPatch(pos[k]-u*(r+0.08), u*1.33, arrowstyle='<|-|>',
                                 mutation_scale=20, color=bs.GRAY, lw=1.8, ls='-',
                                 shrinkA=0, shrinkB=0, zorder=1))

# arc labels, placed outside the ring (never on the arcs)
ax.text(4.35, 2.95, 'accelerated\n(AI / high-throughput)', ha='left', va='center',
        fontsize=18, color='#9A6B00', fontweight='bold', linespacing=1.1)
ax.text(0, -4.55, 'historical bottleneck\n(synthesis and measurement)', ha='center', va='center',
        fontsize=18, color=bs.RED, fontweight='bold', linespacing=1.1)
ax.text(-4.35, 2.95, 'data feedback\n(active learning)', ha='right', va='center',
        fontsize=18, color=bs.GRAY, fontweight='bold', linespacing=1.1)

fig.savefig('../../ch03/figures/Fig3.2_design_make_test.png')
