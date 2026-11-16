"""Ex 7.7.5: cutoff rc = 2.0 A vs 5.0 A for the SchNet-like and NequIP-like models, N=300, 1200 steps, lr 1e-2, batch 8
(same budget/protocol as Lab 7.21). Seeds given on the command line (default 0)."""
import os, sys, json, time, importlib.util, numpy as np, torch
torch.set_num_threads(1)
L = "../ch07/labs/"
_s = importlib.util.spec_from_file_location("m77", L + "77_models.py"); M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
R3, E3, F3, Z = M.load(300)
x = R3[2200].double().numpy(); D = np.linalg.norm(x[:, None] - x[None], axis=-1)
names = ["C1", "C2", "O", "Ho", "H1", "H2", "H3", "H4", "H5"]
print("pairs with r < 2.0 A (frame 2200):", sum(D[i, j] < 2.0 for i in range(9) for j in range(i + 1, 9)), "of 36")
for i in range(9):
    print("  ", names[i], " ".join(f"{names[j]}:{D[i,j]:.2f}" for j in range(9) if j != i))
OrigW = M.Wrapped
RC = float(os.environ.get("RC", "2.0"))
class WrappedRC(OrigW):
    def __init__(self, kind, Z, mu=0.0, sigma=1.0):
        super().__init__(kind, Z, mu, sigma)
        if kind == "schnet": self.core = M.SchNetLike(rc=RC)
        elif kind == "nequip": self.core = M.EquivariantPotential(irreps_hidden="16x0e + 8x1o + 8x1e", lmax=1, n_layers=2, rc=RC)
M.Wrapped = WrappedRC
perm = np.random.default_rng(0).permutation(2000); val = slice(2000, 2100); test = slice(2200, 3000)
seeds = [int(a) for a in sys.argv[1:]] or [0]
for seed in seeds:
    for kind in ("schnet", "nequip"):
        tr = torch.as_tensor(perm[:300]); c0 = time.process_time()
        m = M.train(kind, R3[tr], E3[tr], F3[tr], R3[val], E3[val], F3[val], Z, steps=1200, bs=8, lr=1e-2, seed=seed)
        r = M.evaluate(m, R3[test], E3[test], F3[test])
        print(f"rc={RC} seed {seed} {kind}: test E MAE {r['E_mae']*1e3:.1f} meV, F MAE {r['F_mae']*1e3:.1f} meV/A  (cpu {time.process_time()-c0:.0f} s)", flush=True)
