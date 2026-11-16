"""Ex 8.4.6 positive control: add k labelled 1000 K frames (not in the 1000 K test subset) to the N=300 set and
fine-tune the same three Adam models exactly as in the DAgger/random arms (one fine-tune of 300 steps)."""
import os, time, numpy as np, torch
from common import lab, LABS
torch.set_num_threads(1)
exec(open("ex846_mlip_dagger.py").read().split("arm(\"dagger\")")[0].split("xtb = TBLite")[0])
exec("def finetune" + open("ex846_mlip_dagger.py").read().split("def finetune")[1].split("def arm")[0])
pool = np.setdiff1d(np.arange(1000), S.i10); rng = np.random.default_rng(0)
for k in (5, 50):
    j = torch.as_tensor(rng.choice(pool, k, replace=False))
    R, E, F = torch.cat([S.Rtr, S.R10[j]]), torch.cat([S.Etr, S.E10[j]]), torch.cat([S.Ftr, S.F10[j]])
    ms = [finetune(m, R, E, F, s) for s, m in enumerate(load_models())]
    print(f"+{k} frames from 1000 K: F1000 MAE {f1000(ms):.0f} meV/A | F300 MAE {1e3*np.mean([S.evaluate(m, S.R3[S.test], S.E3[S.test], S.F3[S.test])[1] for m in ms]):.0f} meV/A", flush=True)
ms = [finetune(m, S.Rtr, S.Etr, S.Ftr, s) for s, m in enumerate(load_models())]
print(f"+0 frames (fine-tune only): F1000 MAE {f1000(ms):.0f} meV/A | F300 {1e3*np.mean([S.evaluate(m, S.R3[S.test], S.E3[S.test], S.F3[S.test])[1] for m in ms]):.0f}")
