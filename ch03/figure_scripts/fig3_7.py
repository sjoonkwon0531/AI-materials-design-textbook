"""Fig 3.7 - US vs Korea materials/semiconductor data and industry policy timeline, 2010-2026.
Only milestones stated in ch3.md (Sec. 3.2.1-3.2.4, 3.7.1-3.7.4); positions use the month
where the text gives one, otherwise mid-year.
Staircase layout: events sorted by date, one row each; each label ends at (or starts from)
its own leader line, so labels never overlap and leaders never cross a label."""
import sys
sys.path.insert(0, '../ch02/figures')
import bookstyle as bs
import matplotlib.pyplot as plt

US_Y, KR_Y = 0.0, -1.25
ROW0, DROW = 0.38, 0.37
FS = 17
# (x, label, side)  side: 'L' = label to the left of its leader, 'R' = to the right
US = [(2011.46, 'MGI launched; Materials Project (open DFT database)', 'R'),
      (2021.85, 'MGI Strategic Plan 2021', 'L'),
      (2022.55, 'CHIPS and Science Act', 'L'),
      (2024.44, 'AMII workshop (autonomous materials innovation)', 'L'),
      (2025.90, 'Genesis Mission (AI for science)', 'L'),
      (2026.46, 'CHIPS R&D: SandboxAQ, $500M for AI materials', 'L')]
KR = [(2019.54, 'Japan export controls (sobujang shock)', 'L'),
      (2021.40, 'K-Semiconductor Strategy', 'L'),
      (2022.38, 'K-MDS / NCMRD (national materials data)', 'L'),
      (2023.20, 'Japan lifts controls; white list restored', 'L'),
      (2023.50, 'K-CHIPS Act (tax credit)', 'L'),
      (2025.15, 'K-CHIPS Act amended', 'L'),
      (2026.08, 'Semiconductor Special Act', 'L'),
      (2026.33, 'AI Materials R&D Platform Strategy', 'L')]

top = US_Y + ROW0 + DROW*(len(US) - 1) + 0.3
bot = KR_Y - ROW0 - DROW*(len(KR) - 1) - 0.3
fig, ax = plt.subplots(figsize=(17, 1.02*(top - bot) + 1.3))
X0, X1 = 2008.4, 2027.2
ax.set_xlim(X0, X1); ax.set_ylim(bot, top)
for yr in range(2010, 2027, 2):
    ax.axvline(yr, color='#E6E6E6', lw=1, zorder=0)

def lane(y, items, color, marker, sign):
    ax.plot([2009.9, 2027.0], [y, y], color=color, lw=4.5, zorder=1, solid_capstyle='butt')
    for k, (x, lab, side) in enumerate(items):
        yr = y + sign*(ROW0 + DROW*k)
        ax.plot([x, x], [y, yr], color=color, lw=1.6, zorder=1)
        ax.plot(x, yr, 'o', ms=5, color=color, zorder=2)
        ax.plot(x, y, marker, ms=15, mfc=color, mec='white', mew=2, zorder=3)
        dx = -0.09 if side == 'L' else 0.09
        ax.text(x + dx, yr, lab, ha='right' if side == 'L' else 'left', va='center',
                fontsize=FS, color='#1A1A1A', zorder=4,
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.9))

lane(US_Y, US, bs.NAVY, 'o', +1)
lane(KR_Y, KR, '#B07D00', 's', -1)
ax.text(2009.75, US_Y, 'United\nStates', ha='right', va='center', fontsize=21,
        fontweight='bold', color=bs.NAVY, linespacing=1.0)
ax.text(2009.75, KR_Y, 'Korea', ha='right', va='center', fontsize=21,
        fontweight='bold', color='#8A5A00')

# data-infrastructure gap: Materials Project (2011) -> K-MDS (2022)
gy = (US_Y + KR_Y) / 2
ax.annotate('', xy=(2022.3, gy), xytext=(2011.6, gy),
            arrowprops=dict(arrowstyle='<|-|>', color=bs.GRAY, lw=2, mutation_scale=20))
ax.text(2016.9, gy + 0.05, '~11-year gap in data infrastructure', ha='center',
        va='bottom', fontsize=18, style='italic', color=bs.GRAY)

ax.set_yticks([])
ax.set_xticks(range(2010, 2027, 2))
ax.spines[['left', 'top', 'right']].set_visible(False)
ax.set_xlabel('year')
fig.savefig('../ch03/figures/Fig3.7_KR_timeline.png')
