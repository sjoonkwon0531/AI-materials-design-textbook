"""Ex 7.4.5 (a)(b): reduced extension of Lab 7.16. Arms: scratch 2300 steps vs pretrained (S, G; states from the Lab) + 800 FT steps
(equal total optimisation steps). N = 10, 25, fine-tune subsets seeds 0,1,2. Test: 300 K frames 2000-2999 and
1000 K frames not used in pretraining (odd indices, 500 frames)."""
import time, json, numpy as np, torch
src = open("../../../ch07/labs/74_pretrain_ethanol.py").read().replace("../../ch07/", "../../../ch07/")  # lab paths are relative to ch07/labs
exec(src.split("# ------------------------------------------------------------ pretraining (once per seed)")[0])
pre = torch.load("../../../ch07/labs/74_pretrained_states.pt")
t1k = np.load(DATA + "ethanol_xtb_1000K.npz")
odd = np.arange(1, 1000, 2)
R1k = torch.tensor(t1k["R"][odd], dtype=torch.float32); E1k = torch.tensor(t1k["E"][odd], dtype=torch.float32)
F1k = torch.tensor(t1k["F"][odd], dtype=torch.float32)
print(f"1000 K test (odd frames): {len(odd)}, |F| mean {F1k.abs().mean()*1e3:.0f} meV/A")
def ev1k(m):   # energies: remove mean offset? no - same GFN2 level; report raw MAE
    return evaluate(m, R1k, E1k, F1k)
res = {}
t0 = time.time()
for N in (10, 25):
    for init in ("0", "S", "G"):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(100 + seed)
            tr = torch.tensor(rng.choice(pool, N, replace=False))
            m = SchNetDense(Z); m.load_state_dict(pre[(init, 0)])
            m.shift = E300[tr].mean().item()
            m.scale = {"0": 9 * F300[tr].std().item(), "S": 9.0, "G": 9 * F1.std().item()}[init]
            train(m, R300[tr], E300[tr], F300[tr], 2300 if init == "0" else 800, 2e-3, 32, seed)
            a = evaluate(m, R300[test], E300[test], F300[test]); b = ev1k(m)
            res[f"{N}_{init}_{seed}"] = a + b
        r = np.array([res[f"{N}_{init}_{s}"] for s in (0, 1, 2)])
        print(f"N={N:3d} {init}: 300K E {r[:,0].mean():6.1f}+-{r[:,0].std(ddof=1):4.1f} F {r[:,1].mean():6.1f}+-{r[:,1].std(ddof=1):4.1f} | "
              f"1000K E {r[:,2].mean():6.1f}+-{r[:,2].std(ddof=1):5.1f} F {r[:,3].mean():6.1f}+-{r[:,3].std(ddof=1):5.1f}  "
              f"F300 seeds {np.round(r[:,1],1).tolist()} F1000 seeds {np.round(r[:,3],1).tolist()} [t={time.time()-t0:.0f}s]", flush=True)
json.dump(res, open("ex745_ab.json", "w"))
