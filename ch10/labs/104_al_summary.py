"""Combine the two seeds of Lab 104_al (data/104_al.json = seed 0, data/104_al_s1.json = seed 1)."""
import json, numpy as np
A = [json.load(open('../data/104_al.json')), json.load(open('../data/104_al_s1.json'))]
print("labels | force RMSE 2500 K, mean of 2 seeds (meV/A) | stable MD steps at 2500 K, mean of 4 runs (min-max)")
for r in range(6):
    row = []
    for arm in ('qbc', 'rand', 'aimd'):
        F = np.mean([a[arm][r]['F_rmse'] for a in A]); st = np.array(sum([a[arm][r]['stable_steps'] for a in A], []))
        row.append(f"{arm} {F:4.0f}, {st.mean():3.0f} ({st.min()}-{st.max()})")
    print(f"  {A[0]['qbc'][r]['labels']:2d}  | " + " | ".join(row))
for arm in ('qbc', 'rand', 'aimd'):
    st = np.array([sum([a[arm][r]['stable_steps'] for a in A], []) for r in (3, 4, 5)])
    print(f"{arm}: rounds with 22-34 labels, {int((st == 500).sum())} of {st.size} MD runs reached 500 steps; mean {st.mean():.0f} steps")
