"""Lab 7.2.3 (key 72_schnet): minimal SchNet-like GNN vs ACSF + element-wise MLP (BPNN, Ch.6 style)
on GFN2-xTB ethanol (300 K).  Both are trained on energies AND forces (forces by autograd) with the same
optimiser, steps and data; test = later 800 frames of the trajectory (leakage-aware split).
Runtime: ~25-45 min on one shared CPU thread under load from other jobs (about half that on an idle core)."""
import time, json
import numpy as np, torch
from importlib.machinery import SourceFileLoader
torch.set_num_threads(1)
U = SourceFileLoader("u72b", "../ch07/labs/72b_utils.py").load_module()
OUT = "../ch07/labs"

R, Z, E, F = U.load(300)
tr, va, te = U.split_300K(seed=0)
print(f"ethanol 300 K: {len(E)} frames, Z={Z.tolist()}, E std {E.std()*1000:.0f} meV, "
      f"<|F_comp|> {np.abs(F).mean()*1000:.0f} meV/A")
print(f"split: train pool {len(tr)}, val {len(va)}, test {len(te)} (frames 2200-2999, after a 2 ps gap)")
print(f"baseline (predict train mean E, zero F): E MAE {np.abs(E[te]-E[tr].mean()).mean()*1000:.1f} meV, "
      f"F MAE {np.abs(F[te]).mean()*1000:.1f} meV/A")

STEPS, BS = 4000, 16
res = {}
t_all = time.time()
for N in [100, 300, 1000]:
    idx = tr[:N]
    for name in ["BPNN(ACSF)", "SchNet"]:
        torch.manual_seed(0)
        if name == "SchNet":
            m = U.SchNetMol(Z, n_feat=32, n_rbf=20, cutoff=5.0, n_inter=3)
        else:
            m = U.BPNN(Z, hidden=48)
            m.fit_scaler(torch.as_tensor(R[idx]))
        t = U.train_ef(m, R[idx], E[idx], F[idx], R[va], E[va], F[va], steps=STEPS, bs=BS, lr=3e-3)
        eE, eF = U.evaluate(m, R[te], E[te], F[te])
        res[(name, N)] = (eE * 1000, eF * 1000)
        print(f"N={N:5d} {name:11s} params {U.sn.count_params(m):6d}  test E MAE {eE*1000:6.1f} meV  "
              f"F MAE {eF*1000:6.1f} meV/A   ({t:.0f} s)")
        if name == "SchNet":
            torch.save(m.state_dict(), f"{OUT}/72b_schnet_ethanol_N{N}.pt")
        if N == 1000:
            Ep, Fp = U.ef(m, torch.as_tensor(R[te]))
            np.savez(f"{OUT}/72b_parity_{'schnet' if name=='SchNet' else 'bpnn'}.npz",
                     E=E[te], Ep=Ep.detach().double().numpy(), F=F[te], Fp=Fp.detach().numpy())

for name in ["BPNN(ACSF)", "SchNet"]:
    Ns = np.array([100, 300, 1000]); fm = np.array([res[(name, n)][1] for n in Ns])
    print(f"{name:11s} force learning-curve slope d log MAE / d log N = {np.polyfit(np.log(Ns), np.log(fm), 1)[0]:.2f}")
json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(f"{OUT}/72b_mlip_results.json", "w"), indent=1)
print(f"wall {time.time()-t_all:.0f} s")
