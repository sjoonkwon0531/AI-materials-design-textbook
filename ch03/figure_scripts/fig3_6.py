"""Fig 3.6 - Li-O formation-energy convex hull (values from Lab 3.3, pymatgen)."""
import sys
sys.path.insert(0, '../ch02/figures')
import bookstyle as bs
import matplotlib.pyplot as plt

hull = [(0, 0.0, 'Li'), (1/3, -2.05, 'Li$_2$O'), (0.5, -1.58, 'Li$_2$O$_2$'), (1, 0.0, 'O$_2$')]
xm, ym, eh = 2/3, -0.90, 0.153          # LiO2, metastable
yh = ym - eh                            # hull energy at x = 2/3 (-1.053)

fig, ax = plt.subplots(figsize=(10, 7))
ax.axhline(0, color=bs.GRAY, lw=1, ls='--', zorder=1)
ax.plot([p[0] for p in hull], [p[1] for p in hull], '-', color=bs.NAVY, lw=3.2, zorder=2)
ax.plot([p[0] for p in hull], [p[1] for p in hull], 'o', ms=15, mfc=bs.NAVY, mec='white',
        mew=2, zorder=4)
# metastable LiO2: open square, dotted drop line to the hull
ax.plot([xm, xm], [yh, ym], ':', color=bs.RED, lw=2.8, zorder=3)
ax.plot(xm, ym, 's', ms=11, mfc='white', mec=bs.RED, mew=2.5, zorder=5)

# labels (offset away from lines)
lab = {'Li': (0, 0.07, 'center', 'bottom'), 'Li$_2$O': (0, -0.10, 'center', 'top'),
       'Li$_2$O$_2$': (-0.03, 0.08, 'right', 'bottom'), 'O$_2$': (0, 0.07, 'center', 'bottom')}
for x, y, n in hull:
    dx, dy, ha, va = lab[n]
    ax.text(x + dx, y + dy, n, ha=ha, va=va, fontsize=21, fontweight='bold', color=bs.NAVY)
ax.text(xm, ym + 0.06, 'LiO$_2$', ha='center', va='bottom', fontsize=21,
        fontweight='bold', color=bs.RED)
ax.annotate('$E_{\\mathrm{hull}}$ = +0.153 eV/atom\n(metastable)', xy=(xm - 0.008, (ym + yh)/2),
            xytext=(0.30, -0.66), ha='left', va='center', fontsize=19, color=bs.RED,
            fontweight='bold', linespacing=1.05,
            arrowprops=dict(arrowstyle='-|>', color=bs.RED, lw=1.6, mutation_scale=18,
                            connectionstyle='arc3,rad=0.2', shrinkB=2))
ax.text(0.74, -1.85, 'convex hull\n(stable phases)', ha='center', fontsize=18,
        color=bs.NAVY, rotation=0)

ax.set_xlim(-0.05, 1.05); ax.set_ylim(-2.35, 0.35)
ax.set_xlabel('composition  $x_{\\mathrm{O}}$  (O atomic fraction)')
ax.set_ylabel('formation energy (eV/atom)')
ax.grid(alpha=0.25)
fig.savefig('../ch03/figures/Fig3.6_convex_hull.png')
