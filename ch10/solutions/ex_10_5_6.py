"""Exercise 10.5.6: break-even of distillation. Measures the CPU cost C of training ONE Lab 10.7 student
(SchNet, 64-atom Si) on the same schedule as the lab (100 epochs on 4 labels, then 40 epochs after each of
5 rounds at 10, 16, 22, 28, 34 labels; warm start), using 34 oracle-labelled 64-atom frames from data/104_pool.npz
and data/104_rattle.npz (the cost depends on the number of frames and epochs, not on which frames).
Then Q* = (n t_U + C) / (t_U - t_S) with t_U = 0.308 s, t_S = 9.5 ms (Lab 10.8) and n = 34.
Run: /tmp/claude-0/v10/bin/python ex_10_5_6.py"""
import sys, time, os
import numpy as np, torch
sys.path.insert(0, '../ch10/labs')
from importlib import import_module
C = import_module('104_common')
torch.set_num_threads(1)

fr = C.load_frames(os.path.join(C.DATA, '104_rattle.npz'))[:4] + C.load_frames(os.path.join(C.DATA, '104_pool.npz'))[::20][:30]
e0 = np.mean([f['E'] / 64 for f in fr[:4]])
t0 = time.process_time()
m = C.new_student(0, e0)
C.train(m, fr[:4], 100, seed=0)
for r, n in enumerate((10, 16, 22, 28, 34), 1):
    C.train(m, fr[:n], 40, seed=10 * r)
Ctr = time.process_time() - t0
print(f"training CPU of one student on the Lab 10.7 schedule (34 labels): {Ctr:.1f} s")
tU, tS, n = 0.308, 0.0095, 34
for label, c in [('one student', Ctr), ('committee of 3', 3 * Ctr)]:
    Q = (n * tU + c) / (tU - tS)
    print(f"{label:15s}: C = {c:6.1f} s, n t_U = {n*tU:.2f} s, Q* = {Q:.0f} force calls"
          f" = {Q:.0f} MD steps = {Q*2/1000:.2f} ps at 2 fs")
# upper bound: the whole Lab 10.7 run (3 strategies, exploration MD, evaluation MD) took 508 s CPU
Q = (n * tU + 508) / (tU - tS)
print(f"upper bound with C = 508 s (whole Lab 10.7): Q* = {Q:.0f}")
# what a 1 ns MD of the 64-atom cell costs with each
steps = 1e6 / 2
print(f"1 ns (5e5 steps at 2 fs): oracle {steps*tU/3600:.1f} h, student {steps*tS/3600:.2f} h + training")
