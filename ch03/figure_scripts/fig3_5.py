"""Fig 3.5 - materials database landscape by scale (log x) and data character (bands).
Values as stated in ch3.md Sec. 3.5.2 (as of Oct 2026)."""
import sys
sys.path.insert(0, '../../ch02/figure_scripts')
import bookstyle as bs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(14, 7.2))
XMIN, XMAX = 3e2, 1e8
ax.set_xscale('log'); ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.5, 2.78)

bands = [(2.1, 0.64, 'Curated computed DBs', '#EEF3FC'),
         (0.95, 0.44, 'FAIR repository', '#FBF3E2'),
         (0.0, 0.44, 'AI-generated candidates', '#FBEAEA')]
for y, h, name, c in bands:
    ax.add_patch(Rectangle((XMIN, y - h), XMAX - XMIN, 2*h, fc=c, ec='none', zorder=0))
    ax.text(3.6e2, y + h - 0.05, name, fontsize=19, fontweight='bold', color=bs.GRAY,
            va='top', ha='left')

def db(x, y, name, note, color, marker, where='above', fx=1.0, ha='center', ms=19):
    """Label block (bold name + note) placed above or below the marker."""
    ax.plot(x, y, marker, ms=ms, mfc=color, mec='white', mew=2, zorder=4)
    if where == 'above':
        t = ax.text(x*fx, y + 0.11, note, ha=ha, va='bottom', fontsize=16, color='#333333',
                    linespacing=1.05)
        ax.annotate(name, xy=(0.5 if ha == 'center' else (0 if ha == 'left' else 1), 1),
                    xycoords=t, ha=ha, va='bottom', fontsize=19, fontweight='bold', color=color)
    else:
        t = ax.annotate(name, xy=(x*fx, y - 0.11), ha=ha, va='top', fontsize=19,
                        fontweight='bold', color=color)
        ax.annotate(note, xy=(0.5 if ha == 'center' else (0 if ha == 'left' else 1), 0),
                    xycoords=t, ha=ha, va='top', fontsize=16, color='#333333', linespacing=1.05)

Y = 2.1
# curated computed (circles), labels alternate above / below to avoid collisions
db(8e4, Y, 'JARVIS-DFT', '~80k; multi-property,\nTBmBJ band gaps', bs.PURPLE, 'o',
   where='above', fx=1.15, ha='right')
db(2e5, Y, 'Materials Project', '200k+ materials\nGGA(+U); r$^2$SCAN (2026)', bs.NAVY, 'o',
   where='below', fx=0.82, ha='left')
db(1.2e6, Y, 'OQMD', '1M+; formation\nenergy, convex hulls', bs.BLUE, 'o',
   where='above', fx=0.85, ha='left')
db(3.5e6, Y, 'AFLOW', 'several M;\nhigh-throughput prototypes', bs.GREEN, 'o',
   where='below', fx=0.85, ha='left')
# repository (square)
ax.plot(1.94e7, 0.95, 's', ms=18, mfc='#8A5A00', mec='white', mew=2, zorder=4)
t = ax.text(1.94e7*0.72, 0.93, '~19.4M uploaded entries\n(~4.35M materials);\nraw, heterogeneous',
            ha='right', va='center', fontsize=16, color='#333333', linespacing=1.05)
ax.annotate('NOMAD', xy=(1, 1), xycoords=t, ha='right', va='bottom', fontsize=19,
            fontweight='bold', color='#8A5A00')

# GNoME: predicted -> stable -> realized, one funnel on a single band
g = [(2.2e6, '2.2M\npredicted', 'D'), (3.81e5, '381k\nstable', 'o'), (736, '736\nrealized', '^')]
ax.plot([p[0] for p in g], [0]*3, color=bs.RED, lw=2.5, ls=(0, (4, 2)), zorder=2)
for x, lab, m in g:
    ax.plot(x, 0, m, ms=18, mfc=bs.RED if m != '^' else 'white', mec=bs.RED, mew=2.5, zorder=4)
    ax.text(x, -0.12, lab, ha='center', va='top', fontsize=16, color='#333333', linespacing=1.05)
ax.text(3.81e5, 0.11, 'GNoME', ha='center', va='bottom', fontsize=19, fontweight='bold', color=bs.RED)

ax.set_yticks([])
ax.spines[['left', 'top', 'right']].set_visible(False)
ax.set_xlabel('approximate number of entries (log scale; order of magnitude, as of Oct 2026)')
ax.text(0.995, 0.012, 'sizes change over time (cf. Box 3.6)', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=15, style='italic', color=bs.GRAY)
fig.savefig('../../ch03/figures/Fig3.5_db_landscape.png')
