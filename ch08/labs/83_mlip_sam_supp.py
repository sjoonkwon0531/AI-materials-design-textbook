"""Lab 8.3 (c) supplement: (i) SAM with a rho below the selection grid (0.003), since the grid optimum 0.01 sat
on its lower edge; (ii) Adam with 2400 steps = the same number of gradient evaluations as SAM's 1200 steps.
Usage: python 83_mlip_sam_supp.py   (~6 min on one thread)"""
import os, time, json, importlib.util
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("m83", os.path.join(HERE, "83_mlip_sam.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
res = {}
for name, kw in (("SAM rho=0.003", dict(method="sam", rho=0.003)), ("Adam 2400 steps", dict(method="adam", steps=2400))):
    rows = []
    for seed in (0, 1, 2):
        mod = M.train(seed=seed, **kw)
        _, f3 = M.evaluate(mod, M.R3[M.test], M.E3[M.test], M.F3[M.test])
        _, f6 = M.evaluate(mod, M.R6[M.i6], M.E6[M.i6], M.F6[M.i6]); _, f10 = M.evaluate(mod, M.R10[M.i10], M.E10[M.i10], M.F10[M.i10])
        rows.append(dict(F300=f3, F600=f6, F1000=f10, lam=M.lam(mod)))
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}; res[name] = rows
    print("%-16s F300 %4.0f+/-%2.0f  F600 %4.0f+/-%2.0f  F1000 %4.0f+/-%2.0f meV/A  lam_max %.0f+/-%.0f" % (
        name, 1e3 * A["F300"].mean(), 1e3 * A["F300"].std(), 1e3 * A["F600"].mean(), 1e3 * A["F600"].std(),
        1e3 * A["F1000"].mean(), 1e3 * A["F1000"].std(), A["lam"].mean(), A["lam"].std()), flush=True)
json.dump(res, open(os.path.join(HERE, "83_mlip_supp_results.json"), "w"))
