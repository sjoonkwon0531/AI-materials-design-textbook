"""Lab 9.4-B (part 2b): the long budget.  All 224 target trajectories, 3000 steps (the pre-training budget):
does pre-training still help when target data and compute are plentiful?  Run: python 94c_long.py  (about 4 CPU-min)"""
import sys, time, importlib, numpy as np, torch
sys.path.insert(0, "../ch09/labs")
FS = importlib.import_module("94c_fewshot"); C, P = FS.C, FS.P
pool, te = P.load("react_train"), P.load("react_test")
sub = dict(u=pool["u"][:224], coef=pool["coef"][:224])
for meth, lr in (("scratch", 3e-3), ("scratch", 1e-3), ("full", 3e-3), ("full", 1e-3), ("lora4", 1e-2), ("lora4", 3e-3)):
    t0 = time.process_time(); m, ntr = FS.adapt(meth, sub, 7, lr, steps=P.STEPS)
    print(f"{meth:8s} K = 224, {P.STEPS} steps, lr {lr:.0e}: target-test error {100 * C.traj_error(m, te).mean().item():5.2f}% "
          f"({time.process_time() - t0:.0f} CPU-s)", flush=True)
