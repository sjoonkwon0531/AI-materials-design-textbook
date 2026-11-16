"""Lab 9.3-1 (continued): ablations of the FNO-1D of 93_fno_burgers.py at s = 64.
(a) mode truncation kmax = 4, 8, 16, 32   (b) appending the grid coordinate as an input channel."""
import os, sys, time, json, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models")
t_all = time.time(); c_all = time.process_time()
d = np.load(os.path.join(HERE, "..", "data", "93_burgers.npz"))
traj = torch.tensor(d["traj"]); NTR, NTE, NF = 1000, 200, traj.shape[-1]
A, U = traj[:, 0], traj[:, -1]
tr, te = slice(0, NTR), slice(2000, 2000 + NTE)
sub = lambda X, s: X[:, None, ::NF // s]
EPOCHS, BATCH = 50, 20
prev = json.load(open(os.path.join(HERE, "93_fno_burgers.json")))


def train(net, s, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    a, u = sub(A[tr], s), sub(U[tr], s)
    return M.fit(net, lambda idx: M.rel_l2(net(a[idx]), u[idx]), NTR, epochs, BATCH, lr=3e-3, seed=seed)


@torch.no_grad()
def test(net, s):
    return M.rel_l2_each(net(sub(A[te], s)), sub(U[te], s))


out = {}
# ---------------- (4) mode truncation ----------------
out["kmax"] = [4, 8, 16, 32]; out["kmax_err"] = []; out["kmax_params"] = []
with torch.no_grad():
    uh = torch.fft.rfft(U[te]).abs().pow(2)
    out["kmax_tail"] = [torch.sqrt(uh[:, k + 1:].sum(1) / uh.sum(1)).mean().item() for k in out["kmax"]]
for k, tail in zip(out["kmax"], out["kmax_tail"]):
    torch.manual_seed(0); net = M.FNO1d(width=32, kmax=k, depth=4)
    if k == 16: e = prev["fno_train64"][1]                    # already trained in 93_fno_burgers.py
    else: train(net, 64); e = test(net, 64).mean().item()
    out["kmax_err"].append(e); out["kmax_params"].append(M.count_params(net))
    print(f"kmax={k:2d}: params {M.count_params(net):7d}  test rel-L2 (s=64) {e:.4f}   [truth content above kmax: {tail:.4f}]")


# ---------------- grid channel ----------------
a64 = sub(A[te], 64); sh = 5
torch.manual_seed(0)
netg = M.FNO1d(width=32, kmax=16, depth=4, grid=True); train(netg, 64)
with torch.no_grad():
    eg = test(netg, 64).mean().item()
    eqg = M.rel_l2(netg(torch.roll(a64, sh, -1)), torch.roll(netg(a64), sh, -1)).item()
out["equiv_grid"], out["err_grid"] = eqg, eg
print(f"translation equivariance error (shift {sh}/64): no grid channel {prev['equiv_nogrid']:.1e} (test error {prev['fno_train64'][1]:.4f})"
      f" | with grid channel {eqg:.1e} (test error {eg:.4f})")

json.dump(out, open(os.path.join(HERE, "93_fno_ablation.json"), "w"))
print(f"wall {time.time()-t_all:.0f} s (CPU {time.process_time()-c_all:.0f} s)")
