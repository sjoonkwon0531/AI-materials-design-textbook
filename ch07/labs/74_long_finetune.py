"""Lab 74, part 3: fine-tune the two pretrained models (S: denoising, G: GFN1 labels; saved by
74_pretrain_ethanol.py in 74_pretrained_states.pt) for 2,300 steps, i.e. as long as the scratch control in
74_scratch_control.py, so that all arms are compared at (near) convergence of the fine-tuning stage.
Same model, loss, lr 2e-3 cosine schedule, subsets (seeds 0,1) and test set.
Runtime ~15-30 min under shared load (~5 min on an idle core)."""
import time, numpy as np, torch
src = open("../../ch07/labs/74_pretrain_ethanol.py").read()
exec(src.split("# ------------------------------------------------------------ pretraining (once per seed)")[0])
pre = torch.load("74_pretrained_states.pt")
t0 = time.time()
for N in (50, 200):
    for init in ("S", "G"):
        out = []
        for seed in (0, 1):
            rng = np.random.default_rng(100 + seed)
            tr = torch.tensor(rng.choice(pool, N, replace=False))
            m = SchNetDense(Z); m.load_state_dict(pre[(init, 0)])
            m.shift = E300[tr].mean().item()
            m.scale = {"S": 9.0, "G": 9 * F1.std().item()}[init]
            train(m, R300[tr], E300[tr], F300[tr], 2300, 2e-3, 32, seed)
            out.append(evaluate(m, R300[test], E300[test], F300[test]))
        e = [o[0] for o in out]; f = [o[1] for o in out]
        print(f"N={N:4d} init={init}, 2300 FT steps: E MAE {np.mean(e):6.1f} ({e[0]:.1f}, {e[1]:.1f}) meV   "
              f"F MAE {np.mean(f):6.1f} ({f[0]:.1f}, {f[1]:.1f}) meV/A   [t={time.time()-t0:.0f}s]", flush=True)
