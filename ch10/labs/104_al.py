"""104_al.py -- Lab for Section 10.4.2: query-by-committee active learning with an oracle.

Oracle = MACE-MP-0 small (float32), standing in for DFT.  System = 64-atom diamond Si (periodic).
Students = periodic SchNet (32 features, 2 interactions, r_c = 4 A).
Three arms, same number of oracle LABELS (4 seed + 5 rounds x 6 = 34):
  qbc   : committee of 3 students explores by MD; label the 6 frames with the largest
          max-atom force deviation inside the trust window [0.05, 1.5] eV/A   (DP-GEN style)
  rand  : one student explores the same way; label 6 frames chosen at random among its frames
  aimd  : label 6 random frames of the ORACLE's own MD (needs the oracle trajectory, 200 steps per T)
Exploration temperatures per round: 900, 1500, 2000, 2500, 2500 K (each: 2 student MD runs x 150 steps).
After every round: force RMSE on an independent 2500 K oracle test set (and the 1500 K one) and MD
stability of member 0 at 2500 K (2 runs x 500 steps x 2 fs; failure = Si-Si < 1.9 A or |F| > 25 eV/A).
Seed data: the first 4 rattled crystals of 104_pool.py.  Oracle MD for 'aimd': 104_pool.py + 104_pool_hot.py.
"""
import os, sys, time, json, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')

t0 = time.process_time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # 0 = the run shown in the text
orc = C.oracle('float32')
pool = C.load_frames(os.path.join(C.DATA, '104_pool.npz')) + C.load_frames(os.path.join(C.DATA, '104_pool_hot.npz'))
rat = C.load_frames(os.path.join(C.DATA, '104_rattle.npz'))
test = C.load_frames(os.path.join(C.DATA, '104_test_2500K.npz'))
test1500 = C.load_frames(os.path.join(C.DATA, '104_test.npz'))
base = C.si64(); Z = base.numbers; cell = base.cell.array; x0 = base.positions.copy()
e0 = np.mean([f['E'] / 64 for f in rat])
TEMPS = [900, 1500, 2000, 2500, 2500]; T_TARGET, MD_STEPS = 2500, 500; K = 6; LO, HI = 0.05, 1.5
EPOCHS0, EPOCHS = 100, 40


def stability(m, tag):
    steps = []
    for s in range(2):
        _, fail = C.langevin(m, x0, cell, Z, T_TARGET, MD_STEPS, seed=900 + s, every=10 ** 6)
        steps.append(MD_STEPS if fail is None else fail)
    return steps


def explore(models, T, rnd, committee):
    cand = []
    global fails
    fails = []
    for tr in range(2):
        fr, fail = C.langevin(models[0], x0, cell, Z, T, 150, seed=1000 * rnd + tr + 7 * SEED, every=1,
                              committee=models if committee else None, keep_fail=True)
        for f in fr:
            f['traj'] = tr
        cand += fr
        fails.append(fail)
    return cand


arms = {}
rng_sel = np.random.default_rng(5 + SEED)
for arm in ('qbc', 'rand', 'aimd'):
    data = [dict(f) for f in rat[:4]]
    models = [C.new_student(s + 100 * SEED, e0) for s in (range(3) if arm == 'qbc' else [0])]
    for k, m in enumerate(models):
        C.train(m, data, EPOCHS0, seed=k + 100 * SEED)
    hist = []
    oracle_calls = len(data)
    oracle_md_T = set()
    for rnd in range(len(TEMPS) + 1):
        err = C.errors(models[0], test); err15 = C.errors(models[0], test1500)
        st = stability(models[0], arm)
        hist.append(dict(round=rnd, labels=len(data), F_rmse=err[1], E_rmse=err[0], F_rmse_1500=err15[1], stable_steps=st,
                         oracle_calls=oracle_calls + 200 * len(oracle_md_T)))
        if rnd == len(TEMPS):
            break
        T = TEMPS[rnd]
        if arm == 'aimd':
            oracle_md_T |= {t for t in (300, 900, 1500, 2000, 2500) if t <= T}
            idx = [i for i, f in enumerate(pool) if f['T'] in oracle_md_T and f['step'] > 20]
            new = [dict(pool[i]) for i in rng_sel.choice(idx, K, replace=False)]
            info = f"oracle-MD frames at T in {sorted(oracle_md_T)}"
        else:
            cand = explore(models, T, rnd, committee=(arm == 'qbc'))
            if arm == 'qbc':
                dev = np.array([c['dev'] for c in cand])
                ok = np.where((dev >= LO) & (dev < HI))[0]
                order = ok[np.argsort(-dev[ok])] if len(ok) else np.argsort(-dev)
                pick = []
                for i in order:                        # greedy, at least 5 steps apart within a trajectory
                    if all(cand[i]['traj'] != cand[j]['traj'] or abs(cand[i]['step'] - cand[j]['step']) >= 5 for j in pick):
                        pick.append(i)
                    if len(pick) == K:
                        break
                pick = np.array(pick)
                info = (f"explore MD failed at steps {fails}; {len(cand)} candidates, dev median {np.median(dev):.3f} max {dev.max():.3f} eV/A, "
                        f"{len(ok)} in window, picked dev {dev[pick].min():.3f}-{dev[pick].max():.3f}")
            else:
                pick = rng_sel.choice(len(cand), min(K, len(cand)), replace=False)
                info = f"explore MD failed at steps {fails}; {len(cand)} candidates, random pick"
            new = C.label(orc, [cand[i] for i in pick])
            oracle_calls += len(new)
        data += new          # (aimd: the labels are part of the oracle MD steps counted in oracle_calls)
        for k, m in enumerate(models):
            C.train(m, data, EPOCHS, seed=10 * rnd + k + 100 * SEED)
        print(f"[{arm}] round {rnd + 1} T={T} K: {info}; labels now {len(data)}")
    arms[arm] = hist
    if SEED == 0:
        torch.save(models[0].state_dict(), os.path.join(C.DATA, f'104_al_{arm}.pt'))
    print(f"[{arm}] " + " | ".join(f"n={h['labels']:2d}: F {h['F_rmse']:5.0f} ({h['F_rmse_1500']:4.0f}) meV/A, stable {h['stable_steps']}" for h in hist))

print("summary at equal labels (force RMSE on 2500 K test, meV/A; MD steps survived at 2500 K out of 500, two runs):")
for rnd in range(len(TEMPS) + 1):
    row = " | ".join(f"{a}: {arms[a][rnd]['F_rmse']:5.0f}, {arms[a][rnd]['stable_steps']}" for a in arms)
    print(f"  labels {arms['qbc'][rnd]['labels']:2d} -> {row}")
print("oracle calls spent by the end: " + ", ".join(f"{a} {arms[a][-1]['oracle_calls']}" for a in arms))
json.dump(arms, open(os.path.join(C.DATA, '104_al.json' if SEED == 0 else f'104_al_s{SEED}.json'), 'w'), indent=1)
print(f"CPU {time.process_time() - t0:.0f} s")
