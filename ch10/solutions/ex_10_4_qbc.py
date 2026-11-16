"""Shared driver for Ex 10.4.5 and 10.4.6: the qbc arm of Lab 10.7 (104_al.py) with options.
Usage: python ex_10_4_qbc.py LO SEED VARIANT [M]
  LO      lower trust threshold sigma_lo (eV/A); HI = 1.5 as in the lab
  SEED    as in 104_al.py (0 = the run in the text)
  VARIANT 'base'  : exploration exactly as in Lab 10.7 (perfect 64-atom cell, 2 MD runs per round)
          'mixed' : Ex 10.4.6, second MD run of each round uses a strained (+-3 %) or a vacancy (63-atom) cell
  M       committee size (default 3)
DP-GEN semantics: only frames inside [LO, HI) are labelled (if fewer than 6 are in the window, fewer labels).
(The lab script falls back to the largest-deviation frames when the window is empty; here we do not.)
Evaluation only after the last round (2500 K test RMSE, 1500 K test RMSE, 2 x 500-step 2500 K MD).
"""
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, '../../ch10/labs')
from importlib import import_module
C = import_module('104_common')
torch.set_num_threads(1)
LO, SEED, VARIANT = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MC = int(sys.argv[4]) if len(sys.argv) > 4 else 3
HI = 1.5; K = 6; TEMPS = [900, 1500, 2000, 2500, 2500]; EPOCHS0, EPOCHS = 100, 40
t0 = time.process_time()
orc = C.oracle('float32')
rat = C.load_frames(os.path.join(C.DATA, '104_rattle.npz'))
test = C.load_frames(os.path.join(C.DATA, '104_test_2500K.npz'))
test1500 = C.load_frames(os.path.join(C.DATA, '104_test.npz'))
base = C.si64(); Z = base.numbers; cell = base.cell.array; x0 = base.positions.copy()
e0 = np.mean([f['E'] / 64 for f in rat])


def start(kind):
    if kind == 'perfect':
        return x0.copy(), cell.copy(), Z.copy()
    if kind.startswith('strain'):
        s = 1 + float(kind[6:])
        return x0 * s, cell * s, Z.copy()
    if kind == 'vacancy':
        return x0[1:].copy(), cell.copy(), Z[1:].copy()


PLAN = {'base': [('perfect', 'perfect')] * 5,
        'mixed': [('perfect', 'strain+0.03'), ('perfect', 'vacancy'), ('perfect', 'strain-0.03'),
                  ('perfect', 'vacancy'), ('perfect', 'strain+0.03')]}[VARIANT]
data = [dict(f) for f in rat[:4]]
models = [C.new_student(s + 100 * SEED, e0) for s in range(MC)]
for k, m in enumerate(models):
    C.train(m, data, EPOCHS0, seed=k + 100 * SEED)
log = []
for rnd, T in enumerate(TEMPS):
    cand, fails = [], []
    for tr in range(2):
        xs, cs, zs = start(PLAN[rnd][tr])
        fr, fail = C.langevin(models[0], xs, cs, zs, T, 150, seed=1000 * rnd + tr + 7 * SEED, every=1,
                              committee=models, keep_fail=True)
        for f in fr:
            f['traj'] = tr
        cand += fr; fails.append(fail)
    dev = np.array([c['dev'] for c in cand])
    ok = np.where((dev >= LO) & (dev < HI))[0]
    order = ok[np.argsort(-dev[ok])]
    pick = []
    for i in order:
        if all(cand[i]['traj'] != cand[j]['traj'] or abs(cand[i]['step'] - cand[j]['step']) >= 5 for j in pick):
            pick.append(i)
        if len(pick) == K:
            break
    new = C.label(orc, [cand[i] for i in pick]) if pick else []
    data += new
    for k, m in enumerate(models):
        C.train(m, data, EPOCHS, seed=10 * rnd + k + 100 * SEED)
    kinds = [PLAN[rnd][cand[i]['traj']] for i in pick]
    log.append(dict(T=T, n_cand=len(cand), n_window=len(ok), n_new=len(new), dev_med=float(np.median(dev)),
                    dev_max=float(dev.max()), fails=fails, kinds=kinds))
    print(f"round {rnd+1} T={T} K {PLAN[rnd]}: explore fails {fails}; {len(cand)} candidates, dev median {np.median(dev):.3f} "
          f"max {dev.max():.3f}; in window [{LO},{HI}): {len(ok)}; labelled {len(new)} {kinds}; labels now {len(data)}", flush=True)
err = C.errors(models[0], test); err15 = C.errors(models[0], test1500)
st = []
for s in range(2):
    _, fail = C.langevin(models[0], x0, cell, Z, 2500, 500, seed=900 + s, every=10 ** 6)
    st.append(500 if fail is None else fail)
tag = f"LO{LO}_S{SEED}_{VARIANT}_M{MC}"
print(f"[{tag}] final: labels {len(data)} (oracle calls {len(data)}), F-RMSE 2500 K {err[1]:.0f} meV/A (1500 K {err15[1]:.0f}), "
      f"2500 K MD steps survived {st}; CPU {time.process_time()-t0:.0f} s")
torch.save(models[0].state_dict(), os.path.join('../../ch10/solutions', f'ex_10_4_{tag}.pt'))
json.dump(dict(log=log, F=err[1], F1500=err15[1], stable=st, labels=len(data)),
          open(os.path.join('../../ch10/solutions', f'ex_10_4_{tag}.json'), 'w'))
