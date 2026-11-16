"""Exercise 10.5.5: replay during full fine-tuning (Lab 10.9 extension).
Replay set: 4 rattled cells each of C, SiC, Ge, Al, NaCl (seed 11, disjoint from the forgetting probes, which
use seed 3), labelled by the ORIGINAL MACE-MP-0 small. Every Adam step uses 4 xTB cells + 2 replay cells,
loss = L_xTB + lambda * L_replay (same form: 100 (dE/N)^2 + |dF|^2). Replay energies are shifted by the E0(Si)
re-fit times the number of Si atoms, so that the E0 step does not fight the replay term.
Budget as in the lab: 40 Adam steps, lr 1e-3 cosine, N = 48 xTB cells.
Run: $MLIP_PY_A (MACE/CHGNet environment) ex_10_5_5.py"""
import time, copy, sys
import numpy as np, torch
from ase.build import bulk
import ft_common as FT

torch.set_num_threads(1)
t0 = time.process_time()
base, tr = FT.base, FT.train_all
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODE = sys.argv[2] if len(sys.argv) > 2 else 'random2'   # 'random2': 2 random replay cells per step; 'strat5': one cell of each material per step
LAMS = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.0, 0.3, 1.0, 3.0]

rng = np.random.default_rng(11)
replay = []
for name, a in [('C', bulk('C', 'diamond', a=3.57, cubic=True)), ('SiC', bulk('SiC', 'zincblende', a=4.38, cubic=True)),
                ('Ge', bulk('Ge', 'diamond', a=5.76, cubic=True)), ('Al', bulk('Al', 'fcc', a=4.04, cubic=True).repeat((2, 1, 1))),
                ('NaCl', bulk('NaCl', 'rocksalt', a=5.70, cubic=True))]:
    for k in range(4):
        b = a.copy(); b.positions += rng.normal(0, 0.06, b.positions.shape)
        fr = dict(pos=b.positions.copy(), cell=b.cell.array.copy(), Z=b.numbers.copy())
        E, F = FT.run(base, [fr])
        fr['E'] = E.item(); fr['F'] = F.detach().numpy().astype(np.float64)
        replay.append(fr)


def batch_loss(model, fr, wE=100.0, wF=1.0):
    E, F = FT.run(model, fr, training=True)
    Er = torch.tensor([f['E'] for f in fr]); Fr = torch.tensor(np.concatenate([f['F'] for f in fr]), dtype=torch.float32)
    nat = torch.tensor([len(f['Z']) for f in fr], dtype=torch.float32)
    return wE * ((E - Er) / nat).pow(2).mean() + wF * (F - Fr).pow(2).mean()


def finetune_replay(model, lam, steps=STEPS, lr=1e-3, seed=0):
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)          # the calculator loads the model frozen
    opt = torch.optim.Adam(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    r = np.random.default_rng(seed); r2 = np.random.default_rng(seed + 100)
    for s in range(steps):
        fr = [tr[k] for k in r.choice(len(tr), 4, replace=False)]
        loss = batch_loss(model, fr)
        if MODE == 'strat5':
            rp = [replay[4 * mat + int(r2.integers(4))] for mat in range(5)]
        else:
            rp = [replay[k] for k in r2.choice(len(replay), 2, replace=False)]   # drawn even for lambda = 0 (same xTB batches)
        if lam > 0:
            loss = loss + lam * batch_loss(model, rp)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


print(f"replay set: {len(replay)} cells; forgetting probes: {len(FT.probes)} cells (lab set); {STEPS} Adam steps; replay batches {MODE}")
for lam in LAMS:
    tm = time.process_time()
    m = copy.deepcopy(base)
    d = FT.refit_e0(m, tr)
    nSi = lambda f: int(np.sum(np.asarray(f['Z']) == 14))
    for f in replay:                                   # keep replay energies on the shifted E0(Si) scale
        f['E_orig'] = f.get('E_orig', f['E']); f['E'] = f['E_orig'] + d * nSi(f)
    finetune_replay(m, lam)
    e, f = FT.errors(m, FT.test); a0, B = FT.eos(m); dr = FT.drift(m)
    mean_dr = np.mean([dr[k] for k in ('C', 'SiC', 'Ge', 'Al', 'NaCl')])
    print(f"lambda {lam:3.1f}: xTB test E {e:6.2f} meV/atom, F {f:4.0f} meV/A | a0 {a0:.4f} B {B:5.1f} GPa | drift C/SiC/Ge/Al/NaCl "
          + "/".join(f"{dr[k]:.0f}" for k in ('C', 'SiC', 'Ge', 'Al', 'NaCl')) + f" (mean {mean_dr:.0f}) meV/A | {time.process_time() - tm:.0f} s")
print(f"CPU {time.process_time() - t0:.0f} s")
