"""Ex 7.4.5 (c): bigger SchNet (F=64, 3 interactions). G-pretraining (GFN1 labels, 1500 steps, seed 0), then
scratch 2300 steps vs G + 800 FT, N=50 (scratch only, seeds 0,1) and N=200 (seeds 0,1,2)."""
import time, json, numpy as np, torch
src = open("../ch07/labs/74_pretrain_ethanol.py").read()
exec(src.split("# ------------------------------------------------------------ pretraining (once per seed)")[0])
t0 = time.time()
torch.manual_seed(0); m0 = SchNetDense(Z, F=64, n_inter=3); s0 = {k: v.clone() for k, v in m0.state_dict().items()}
print("params", sum(p.numel() for p in m0.parameters()))
torch.manual_seed(0); mG = SchNetDense(Z, F=64, n_inter=3); set_scale(mG, E1, F1)
train(mG, Rp, E1, F1, 1500, 2e-3, 32, 0); sG = {k: v.clone() for k, v in mG.state_dict().items()}
print(f"G pretrain {time.time()-t0:.0f}s; zero-shot F MAE {evaluate(mG, R300[test], E300[test], F300[test])[1]:.1f}", flush=True)
res = {}
for N, init, seeds in ((50, "0", (0, 1)), (200, "0", (0, 1, 2)), (200, "G", (0, 1, 2))):
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        tr = torch.tensor(rng.choice(pool, N, replace=False))
        m = SchNetDense(Z, F=64, n_inter=3); m.load_state_dict(s0 if init == "0" else sG)
        m.shift = E300[tr].mean().item()
        m.scale = 9 * F300[tr].std().item() if init == "0" else 9 * F1.std().item()
        train(m, R300[tr], E300[tr], F300[tr], 2300 if init == "0" else 800, 2e-3, 32, seed)
        res[f"{N}_{init}_{seed}"] = evaluate(m, R300[test], E300[test], F300[test])
        print(N, init, seed, np.round(res[f"{N}_{init}_{seed}"], 1), f"[t={time.time()-t0:.0f}s]", flush=True)
json.dump(res, open("ex745_c.json", "w"))
