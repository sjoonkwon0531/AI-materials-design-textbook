# Ex 2.6.4: discrete check of dF/dt = -int |grad mu|^2 along the Lab 2.14 trajectory, effect of S and dt
import numpy as np
from ch_common import ch_run, free_energy, grad_mu_sq
N = 256; T = 5.0
print(" S    dt     rel.err of dF/dt vs -int|grad mu|^2 (median over t in [1,5])   F(5)     F-increase steps")
for S in [0.0, 1.0, 2.0, 4.0, 8.0]:
    for dt in [0.05, 0.01, 0.002]:
        rec = {'prev': None, 'err': [], 'inc': 0, 'bad': False}
        def cb(t, c, ch):
            if not np.isfinite(c).all() or np.abs(c).max() > 10: rec['bad'] = True; return False
            F, D = free_energy(c), grad_mu_sq(c)
            if rec['prev'] is not None:
                F0, D0, t0 = rec['prev']
                if F > F0: rec['inc'] += 1
                if t0 >= 1.0: rec['err'].append(abs((F-F0)/(t-t0) + 0.5*(D+D0))/(0.5*(D+D0)))
            rec['prev'] = (F, D, t)
        with np.errstate(all='ignore'):
            ch_run(N=N, dt=dt, S=S, T=T, seed=1, callback=cb, every=1)
        if rec['bad']: print(f"{S:3.0f}  {dt:5.3f}   UNSTABLE (blow-up)"); continue
        print(f"{S:3.0f}  {dt:5.3f}   {np.median(rec['err']):.3e}   (max {np.max(rec['err']):.2e})                       {rec['prev'][0]:.4f}   {rec['inc']}")
