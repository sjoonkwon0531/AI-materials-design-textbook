"""Ex 8.3.6: (a) 3x longer training (3600 steps) Adam vs SAM; (c) add 30 unseen 600 K frames to the training set."""
import os, sys, time, json, importlib.util, numpy as np, torch
torch.set_num_threads(1)
HERE = "../ch08/labs"
_s = importlib.util.spec_from_file_location("m83", os.path.join(HERE, "83_mlip_sam.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
t0 = time.time()
def report(tag, mod, cpu):
    _, f3 = M.evaluate(mod, M.R3[M.test], M.E3[M.test], M.F3[M.test])
    _, f6 = M.evaluate(mod, M.R6[M.i6], M.E6[M.i6], M.F6[M.i6]); _, f10 = M.evaluate(mod, M.R10[M.i10], M.E10[M.i10], M.F10[M.i10])
    lam = M.lam(mod)
    print(f"{tag}: F300 {1e3*f3:5.0f} F600 {1e3*f6:5.0f} F1000 {1e3*f10:5.0f} meV/A | lam_max {lam:8.1f} | cpu {cpu:.0f} s  [{time.time()-t0:.0f} s]", flush=True)
    return dict(F300=f3, F600=f6, F1000=f10, lam=lam)
part = sys.argv[1]; res = {}
if part == "a":
    method, rho = sys.argv[2], float(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
    for sd in seeds:
        c0 = time.process_time(); mod = M.train(method, sd, rho=rho, steps=3600)
        res[sd] = report(f"{method} rho={rho} 3600 steps seed {sd}", mod, time.process_time() - c0)
    json.dump(res, open(f"s83_6_a_{method}_{rho}.json", "w"))
elif part == "c":
    R6, E6, F6 = M.R6, M.E6, M.F6
    rest = np.setdiff1d(np.arange(len(E6)), M.i6)
    add = np.sort(np.random.default_rng(3).choice(rest, 30, replace=False))
    print(f"600 K frames available {len(E6)}, test uses {len(M.i6)}, added 30 from the other {len(rest)}; overlap with test: {len(np.intersect1d(add, M.i6))}")
    tr270 = torch.as_tensor(M.perm[:270]); a = torch.as_tensor(add)
    M.Rtr = torch.cat([M.R3[tr270], R6[a]]); M.Etr = torch.cat([M.E3[tr270], E6[a]]); M.Ftr = torch.cat([M.F3[tr270], F6[a]])
    for sd in (0, 1, 2):
        c0 = time.process_time(); mod = M.train("adam", sd, steps=1200)
        res[sd] = report(f"Adam 270x300K + 30x600K, 1200 steps, seed {sd}", mod, time.process_time() - c0)
    json.dump(res, open("s83_6_c.json", "w"))
print(f"wall {time.time()-t0:.0f} s")
