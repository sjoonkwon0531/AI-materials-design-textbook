"""Ex 7.7.4: O-H (and C-C, C-O) bond-length distributions pooled over several initial-velocity seeds, 2 ps NVE each."""
import os, sys, numpy as np, torch, json
sys.argv = ["x"]
src = open("../ch07/labs/77_md.py").read().split("out = {}")[0]
src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))', 'HERE = "../ch07/labs"')
exec(src)
from tblite.ase import TBLite
plan = {"xTB": range(5), "schnet_N1000": range(5), "nequip_N1000": range(3)}
out = {}
for key, seeds in plan.items():
    for sd in seeds:
        if key == "xTB": calc = TBLite(method="GFN2-xTB", verbosity=0)
        else:
            w = M.Wrapped(key.split("_N")[0], Z); w.load_state_dict(torch.load(os.path.join(HERE, f"77_{key}.pt"))); calc = TorchCalc(w)
        r = run(calc, key, seed=sd)
        out[(key, sd)] = r
        print(key, sd, r["status"], "drift %.2f meV/ps" % (1e3 * r["drift"]), "<T> %.0f K" % np.mean(r["T"]),
              " ".join("%s %.4f" % (b, np.mean(v)) for b, v in r["bonds"].items()), "cpu %.0f s" % r["cpu"], flush=True)
def pool(key, seeds, b): return np.concatenate([out[(key, s)]["bonds"][b] for s in seeds])
res = {}
for b in BONDS:
    noise = wasserstein_distance(pool("xTB", [0, 1], b), pool("xTB", [2, 3, 4], b))
    line = f"{b}: noise floor W1(xTB s0-1 vs s2-4) = {noise:.4f}"
    for key, seeds in (("schnet_N1000", range(5)), ("nequip_N1000", range(3))):
        w_all = wasserstein_distance(pool(key, seeds, b), pool("xTB", range(5), b))
        w_same = wasserstein_distance(pool(key, seeds, b), pool("xTB", seeds, b))
        line += f" | {key}: W1 vs xTB(all 5) {w_all:.4f}, vs xTB(same seeds) {w_same:.4f}"
    wsn = wasserstein_distance(pool("schnet_N1000", range(3), b), pool("nequip_N1000", range(3), b))
    line += f" | schnet vs nequip (seeds 0-2) {wsn:.4f}"
    # single-seed noise: pairwise xTB seeds
    pw = [wasserstein_distance(out[("xTB", i)]["bonds"][b], out[("xTB", j)]["bonds"][b]) for i in range(5) for j in range(i + 1, 5)]
    line += f" | single-seed xTB pairwise W1 median {np.median(pw):.4f} (range {min(pw):.4f}-{max(pw):.4f})"
    print(line)
    for key, seeds in (("xTB", range(5)), ("schnet_N1000", range(5)), ("nequip_N1000", range(3))):
        p = pool(key, seeds, b); print(f"     {key:13s} pooled mean {p.mean():.4f} std {p.std():.4f}")
