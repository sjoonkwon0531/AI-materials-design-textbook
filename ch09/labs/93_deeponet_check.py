"""Lab 9.3-2 (robustness check): is the DeepONet of 93_deeponet.py limited by capacity or training budget?
A 5x larger DeepONet (width 256, depth 4) trained 3.3x longer (20,000 updates) on the same N = 1000 functions."""
import os, sys, time, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models")
d = np.load(os.path.join(HERE, "..", "data", "93_burgers.npz")); traj = torch.tensor(d["traj"])
a, u = traj[:, 0, ::8], traj[:, -1, ::8]; te = slice(2000, 2200); y = (torch.arange(64) / 64)[:, None]
torch.manual_seed(0); net = M.DeepONet(64, p=64, width=256, depth=4, periodic=4)
c0 = time.process_time()
h, wall = M.fit(net, lambda idx: M.rel_l2(net(a[idx], y), u[idx]), 1000, 1000, 50, lr=1e-3, wd=1e-5, seed=0)
with torch.no_grad(): e = M.rel_l2(net(a[te], y), u[te]).item()
print(f"DeepONet width 256, depth 4 ({M.count_params(net)} params), N=1000, 20000 updates: train {h[-1]:.4f}  test {e:.4f}"
      f"   ({wall:.0f} s wall, {time.process_time()-c0:.0f} s CPU)")
