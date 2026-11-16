"""Fig 3.8 - cumulative roadmap: four paradigms stack (Past - Present - Future) and converge.
Paradigm names and colors match Fig 3.1; Kohn-Sham DFT sits in Computation & data (Sec. 3.1.3)."""
import sys
sys.path.insert(0, '../../ch02/figure_scripts')
import bookstyle as bs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(17, 8.4))
ax.set_xlim(0, 13.6); ax.set_ylim(-0.05, 5.35); ax.axis('off')

# time zones (schematic)
zones = [(0, 5.4, 'PAST', '#F2F3F5'), (5.4, 8.9, 'PRESENT', '#FBF4E6'),
         (8.9, 13.6, 'FUTURE', '#E9EFFB')]
for a, b, name, c in zones:
    ax.add_patch(Rectangle((a, -0.05), b - a, 5.4, fc=c, ec='none', zorder=0))
    ax.text((a + b)/2, 5.08, name, ha='center', va='center', fontsize=22,
            fontweight='bold', color='#7A7F8A')

END = 9.75
H = 0.95
bars = [  # (y, start, name, examples, face, text color)
    (0.15, 0.25, 'Trial & error', "Edison's filament; synthesis and testing to this day", '#9EA3AE', '#1A1A1A'),
    (1.3, 2.2, 'Theory-driven', 'band theory; BCS theory', '#6874AE', 'white'),
    (2.45, 3.9, 'Computation & data',
     'Kohn-Sham DFT (1965); MGI (2011);\nMaterials Project; K-MDS (2022)', bs.GOLD, '#1A1A1A'),
    (3.6, 5.8, 'AI-driven', 'GNoME (2023); MatterGen (2025)', bs.NAVY, 'white'),
]
box_x, box_w, box_y, box_h = 10.45, 3.0, 1.15, 2.4
for y, x0, name, ex, fc, tc in bars:
    ax.add_patch(FancyBboxPatch((x0, y), END - x0, H, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=fc, ec='white', lw=2, zorder=2))
    ax.text(x0 + 0.18, y + H - 0.12, name, ha='left', va='top', fontsize=22,
            fontweight='bold', color=tc, zorder=3)
    ax.text(x0 + 0.18, y + 0.1, ex, ha='left', va='bottom', fontsize=17, color=tc,
            style='italic', zorder=3, linespacing=1.0)
    # each paradigm continues into the future and converges
    ax.add_patch(FancyArrowPatch((END + 0.02, y + H/2), (box_x - 0.05, box_y + box_h/2),
                                 arrowstyle='-|>', mutation_scale=22, lw=2.2,
                                 color=bs.NAVY, connectionstyle='arc3,rad=0', zorder=1))

ax.add_patch(FancyBboxPatch((box_x, box_y), box_w, box_h,
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            fc=bs.ICE, ec=bs.NAVY, lw=2.5, zorder=3))
ax.text(box_x + box_w/2, box_y + box_h/2, 'Materials AI\nfoundation models\n+\nself-driving labs\n(SDL)',
        ha='center', va='center', fontsize=21, fontweight='bold', color=bs.NAVY,
        linespacing=1.1, zorder=4)

fig.savefig('../../ch03/figures/Fig3.8_roadmap.png')
