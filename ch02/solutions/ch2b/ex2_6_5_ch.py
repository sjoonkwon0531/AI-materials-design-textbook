# Ex 2.6.5 (companion): Lab 2.14 Cahn-Hilliard run measured with the SAME interface-based length L_int = A/P
import numpy as np
from ch_common import ch_run, length_scale
from ex2_6_5_tools import interface_length_scale
Lbox = 4*np.pi; rec = {}
def cb(t, c, ch):
    if any(abs(t - s) < 1e-9 for s in (5, 10, 20, 40, 80, 160, 320, 400)):
        rec[round(t)] = (interface_length_scale(c, Lbox), length_scale(c, Lbox), c.mean())
ch_run(T=400.0, seed=1, callback=cb, every=20)
m0 = rec[5][2]
print("CH (Lab 2.14):   t    L_int    2pi/<k>    mean c      L_int^2/t   L_int^3/t")
for t in sorted(rec): print(f"              {t:5d}   {rec[t][0]:6.3f}   {rec[t][1]:6.3f}   {rec[t][2]:+.2e}   {rec[t][0]**2/t:.4f}      {rec[t][0]**3/t:.4f}")
for lo, hi in [(10, 160), (20, 400)]:
    tt = np.array([t for t in sorted(rec) if lo <= t <= hi])
    print(f"CH slope {lo}<=t<={hi}: L_int n = {np.polyfit(np.log(tt), np.log([rec[t][0] for t in tt]), 1)[0]:.3f}, "
          f"2pi/<k> n = {np.polyfit(np.log(tt), np.log([rec[t][1] for t in tt]), 1)[0]:.3f}")
