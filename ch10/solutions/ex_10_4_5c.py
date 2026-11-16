"""Ex 10.4.5 (c): distribution of the committee deviation eps_F (Eq. 10.4.1) vs committee size M = 2, 3, 5.
Five students trained on the same 16 frames (4 rattled seed cells + 12 random 900/1500 K oracle-MD frames);
member 0 runs one 150-step 1500 K Langevin MD; eps_F evaluated on each frame with members {0,1}, {0,1,2}, {0..4}.
Also the estimator-level check: E[(1/M) sum_k |F_k - Fbar|^2] = (M-1)/M * 3 s^2 for iid noise."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, '../../ch10/labs')
from importlib import import_module
C = import_module('104_common')
torch.set_num_threads(1)
t0 = time.process_time()
rat = C.load_frames(os.path.join(C.DATA, '104_rattle.npz'))
pool = C.load_frames(os.path.join(C.DATA, '104_pool.npz'))
rng = np.random.default_rng(0)
idx = [i for i, f in enumerate(pool) if f['T'] in (900, 1500) and f['step'] > 20]
data = [dict(f) for f in rat[:4]] + [dict(pool[i]) for i in rng.choice(idx, 12, replace=False)]
e0 = np.mean([f['E'] / 64 for f in rat])
models = []
for s in range(5):
    m = C.new_student(s, e0); C.train(m, data, 100, seed=s); models.append(m)
base = C.si64()
fr, fail = C.langevin(models[0], base.positions, base.cell.array, base.numbers, 1500, 150, seed=3, every=1)
Fs = []
for f in fr:
    Fs.append(np.array([o[1] for o in C.forces_of(models, f['pos'], f['cell'], f['Z'])]))
Fs = np.array(Fs)                                     # (frames, 5, atoms, 3)
for Msub in [2, 3, 5]:
    S = Fs[:, :Msub]
    epsF = np.sqrt(((S - S.mean(1, keepdims=True)) ** 2).sum(-1).mean(1)).max(-1)
    mean_atom = np.sqrt(((S - S.mean(1, keepdims=True)) ** 2).sum(-1).mean(1)).mean(-1)
    fr_lo = {lo: (epsF >= lo).mean() for lo in (0.05, 0.1, 0.2)}
    print(f"M={Msub}: eps_F median {np.median(epsF):.3f} eV/A, 90th pct {np.percentile(epsF,90):.3f}, max {epsF.max():.3f}; "
          f"atom-mean deviation median {np.median(mean_atom):.3f}; fraction >= 0.05/0.1/0.2: "
          + "/".join(f"{fr_lo[k]:.2f}" for k in fr_lo) + f"; frames {len(epsF)}, MD fail {fail}", flush=True)
# all 10 pairs and 10 triples, to show the spread of small committees
from itertools import combinations
for Msub in [2, 3]:
    meds = []
    for c in combinations(range(5), Msub):
        S = Fs[:, list(c)]
        meds.append(np.median(np.sqrt(((S - S.mean(1, keepdims=True)) ** 2).sum(-1).mean(1)).max(-1)))
    print(f"M={Msub}, all {len(meds)} subsets: median eps_F ranges {min(meds):.3f}-{max(meds):.3f} eV/A")
for M in [2, 3, 5]:
    print(f"iid noise: sqrt((M-1)/M) = {np.sqrt((M-1)/M):.3f}")
print(f"CPU {time.process_time()-t0:.0f} s")
