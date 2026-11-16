"""Ch.7.7 supplement: is the N = 1000 ranking a budget effect? Retrain the two GNNs at N = 1000 with 3x the
step budget (6000 steps, batch 8, lr 1e-2, cosine), same split/seed as 77_train.py. Saves 77_<kind>_N1000_long.pt
and adds '<kind>_N1000_long' to 77_results.json.  (~10-12 CPU-min.)"""
import os, sys, json, time, importlib.util
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("m77", os.path.join(HERE, "77_models.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
R3, E3, F3, Z = M.load(300); R6, E6, F6, _ = M.load(600); R10, E10, F10, _ = M.load(1000)
perm = np.random.default_rng(0).permutation(2000); val = slice(2000, 2100); test = slice(2200, 3000)
i6 = np.sort(np.random.default_rng(1).choice(3000, 500, replace=False))
i10 = np.sort(np.random.default_rng(2).choice(1000, 500, replace=False))
tr = torch.as_tensor(perm[:1000]); path = os.path.join(HERE, "77_results.json")
for kind in sys.argv[1:] or ["schnet", "nequip"]:
    c0 = time.process_time(); log = []
    model = M.train(kind, R3[tr], E3[tr], F3[tr], R3[val], E3[val], F3[val], Z, steps=6000, bs=8, lr=1e-2, seed=0, log=log)
    cpu = time.process_time() - c0
    torch.save(model.state_dict(), os.path.join(HERE, f"77_{kind}_N1000_long.pt"))
    r = {"params": sum(p.numel() for p in model.parameters()), "cpu_s": cpu, "log": log}
    for tag, (R, E, F) in {"300K": (R3[test], E3[test], F3[test]), "600K": (R6[i6], E6[i6], F6[i6]),
                           "1000K": (R10[i10], E10[i10], F10[i10])}.items():
        r[tag] = M.evaluate(model, R, E, F)
    res = json.load(open(path)); res[f"{kind}_N1000_long"] = r; json.dump(res, open(path, "w"))
    print("N=1000 %-7s 6000 steps | 300K: E %5.1f meV F %4.0f meV/A | 600K F %4.0f | 1000K F %4.0f | cpu %4.0f s" % (
        kind, 1e3 * r["300K"]["E_mae"], 1e3 * r["300K"]["F_mae"], 1e3 * r["600K"]["F_mae"], 1e3 * r["1000K"]["F_mae"], cpu), flush=True)
