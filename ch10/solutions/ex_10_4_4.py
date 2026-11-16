"""Ex 10.4.4: leakage vs minimum time gap between training and test frames (Lab 10.8 Part A extended).
Usage: python ex_10_4_4.py small|big   (v10 interpreter)
 small: Lab student (32 features, 2 interactions), 120 epochs (reproduces Lab 10.8 Part A training)
 big  : 64 features, 3 interactions, 400 epochs."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, '../../ch10/labs')
from importlib import import_module
C = import_module('104_common')
torch.set_num_threads(1)
size = sys.argv[1] if len(sys.argv) > 1 else 'small'
t0 = time.process_time()
pool = C.load_frames(os.path.join(C.DATA, '104_pool.npz'))
test = C.load_frames(os.path.join(C.DATA, '104_test.npz'))
traj = [f for f in pool if f['T'] == 1500 and f['step'] > 20]
rng = np.random.default_rng(0)
first = np.arange(0, 120)
tr_idx = np.sort(rng.choice(first, 24, replace=False))
e0 = np.mean([f['E'] / 64 for f in traj])
if size == 'small':
    m = C.new_student(0, e0); ep = 120
else:
    torch.manual_seed(0); m = C.sn.SchNet(n_feat=64, n_rbf=16, cutoff=C.RC, n_inter=3); m.e_shift[14] = e0; ep = 400
npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
C.train(m, [traj[k] for k in tr_idx], epochs=ep, seed=0)
gap = np.array([2 * np.abs(tr_idx - k).min() for k in range(len(traj))])      # fs
bins = [(0, 0), (2, 4), (6, 10), (12, 20), (22, 40), (42, 1e9)]
print(f"[{size}] params {npar}, epochs {ep}, training frames at steps {list(tr_idx+21)}")
for lo, hi in bins:
    sel = [k for k in range(len(traj)) if lo <= gap[k] <= hi]
    e, f = C.errors(m, [traj[k] for k in sel])
    lab = 'training frames' if hi == 0 else f'gap {lo:g}-{hi:g} fs' if hi < 1e8 else f'gap >= {lo} fs'
    print(f"  {lab:18s} ({len(sel):3d} frames): F-RMSE {f:6.1f} meV/A, E-RMSE {e:6.2f} meV/atom", flush=True)
e, f = C.errors(m, test)
print(f"  independent 1500 K trajectory ( 21 frames): F-RMSE {f:6.1f} meV/A, E-RMSE {e:6.2f} meV/atom")
print(f"CPU {time.process_time()-t0:.0f} s")
