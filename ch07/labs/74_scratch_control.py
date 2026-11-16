"""Lab 74 control: is the pretraining gain just extra optimisation steps?
Scratch models trained for 2,300 steps (= 1,500 pretraining + 800 fine-tuning steps of the pretrained arms),
same model, loss, lr schedule shape, data subsets and test set as 74_pretrain_ethanol.py.
Runtime ~15-20 min under shared load (~4 min on an idle core)."""
import time, numpy as np, torch
src = open("../ch07/labs/74_pretrain_ethanol.py").read()
exec(src.split("# ------------------------------------------------------------ pretraining (once per seed)")[0])
t0 = time.time()
for N in (50, 200):
    out = []
    for seed in (0, 1):
        rng = np.random.default_rng(100 + seed)
        tr = torch.tensor(rng.choice(pool, N, replace=False))
        torch.manual_seed(0)
        m = SchNetDense(Z)
        m.shift = E300[tr].mean().item(); m.scale = 9 * F300[tr].std().item()
        train(m, R300[tr], E300[tr], F300[tr], 2300, 2e-3, 32, seed)
        out.append(evaluate(m, R300[test], E300[test], F300[test]))
    e = [o[0] for o in out]; f = [o[1] for o in out]
    print(f"N={N:4d} scratch, 2300 steps: E MAE {np.mean(e):6.1f} ({e[0]:.1f}, {e[1]:.1f}) meV   "
          f"F MAE {np.mean(f):6.1f} ({f[0]:.1f}, {f[1]:.1f}) meV/A   [t={time.time()-t0:.0f}s]", flush=True)
