"""104_data.py -- Lab for Section 10.4.1: redundancy of MD frames and four ways to pick 16 labels.

Oracle data from 104_pool.py (MACE-MP-0 small standing in for DFT; 64-atom diamond Si).
Student: periodic SchNet (32 features, 2 interactions, r_c = 4 A) from 104_common.py.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')
from ase import Atoms
from dscribe.descriptors import SOAP

t0 = time.process_time()
pool = C.load_frames(os.path.join(C.DATA, '104_pool.npz'))
rat = C.load_frames(os.path.join(C.DATA, '104_rattle.npz'))
test = C.load_frames(os.path.join(C.DATA, '104_test.npz'))
hot = C.load_frames(os.path.join(C.DATA, '104_transfer_1800K.npz'))
strain = C.load_frames(os.path.join(C.DATA, '104_transfer_strain.npz'))
Ts = np.array([f['T'] for f in pool])

# ---- (1) redundancy: force autocorrelation along each oracle trajectory
print("force autocorrelation <F(t).F(t+lag)>/<F.F> along oracle MD (dt = 2 fs)")
acf = {}
for T in (300, 900, 1500):
    F = np.array([f['F'] for f in pool if f['T'] == T])[50:]           # drop the first 100 fs
    lags = np.arange(0, 41)
    c = [np.mean([(F[t] * F[t + L]).sum() for t in range(len(F) - L)]) for L in lags]
    c = np.array(c) / c[0]
    acf[T] = c
    first = int(np.argmax(c < 0.1))
    print(f"  {T:5d} K: lag 1 = {c[1]:.3f}, lag 5 = {c[5]:.3f}, lag 10 = {c[10]:.3f}; first below 0.1 at lag {first} ({2 * first} fs)")

# ---- (2) SOAP descriptors of every pool frame (cell average), then FPS
soap = SOAP(species=['Si'], r_cut=4.5, n_max=6, l_max=4, sigma=0.5, periodic=True, average='inner')
def desc(frames):
    return soap.create([Atoms(numbers=f['Z'], positions=f['pos'], cell=f['cell'], pbc=True) for f in frames], n_jobs=1)
X = desc(pool); Xr = desc(rat); Xt = desc(test)
mu, sd = X.mean(0), X.std(0) + 1e-12
Z_, Zr, Zt = (X - mu) / sd, (Xr - mu) / sd, (Xt - mu) / sd
print(f"SOAP: {X.shape[1]} components per frame")

def fps(Zs, n, start):
    sel = [start]; d = np.linalg.norm(Zs - Zs[start], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d)); sel.append(k); d = np.minimum(d, np.linalg.norm(Zs - Zs[k], axis=1))
    return sel

def nn_dist(Zsel, Zq):
    return np.array([np.linalg.norm(Zsel - z, axis=1).min() for z in Zq])

N = 16
i1500 = np.where(Ts == 1500)[0]
strategies = {}
for seed in range(3):
    rng = np.random.default_rng(seed)
    st = int(rng.integers(50, len(i1500) - N))
    strategies[('block', seed)] = list(i1500[st:st + N])                    # 16 consecutive 1500 K frames
    strategies[('random', seed)] = list(rng.choice(len(pool), N, replace=False))
    strategies[('fps', seed)] = fps(Z_, N, int(rng.integers(len(pool))))
    strategies[('rattle', seed)] = None                                    # the 16 rattled crystals

e0 = np.mean([f['E'] / len(f['Z']) for f in pool])
res = {}
for (name, seed), idx in strategies.items():
    tr = rat if idx is None else [pool[k] for k in idx]
    Zsel = Zr if idx is None else Z_[idx]
    cover = np.median(nn_dist(Zsel, Zt))
    m = C.new_student(seed, e0)
    C.train(m, tr, epochs=120, seed=seed)
    r = dict(test=C.errors(m, test), hot=C.errors(m, hot), strain=C.errors(m, strain), cover=cover,
             Tmix=sorted(set(int(tr_['T']) for tr_ in tr)) if idx is not None else ['rattle'])
    res.setdefault(name, []).append(r)
    print(f"  {name:6s} seed {seed}: test-1500K F-RMSE {r['test'][1]:6.1f} meV/A | 1800K {r['hot'][1]:6.1f} | strained {r['strain'][1]:6.1f} | E-RMSE {r['test'][0]:5.2f} meV/atom | SOAP NN-dist to test {cover:5.2f}")

print("summary (mean +/- std over 3 seeds), force RMSE in meV/A:")
summ = {}
for name, rs in res.items():
    a = np.array([[r['test'][1], r['hot'][1], r['strain'][1], r['test'][0], r['cover']] for r in rs])
    summ[name] = dict(mean=a.mean(0).tolist(), std=a.std(0).tolist())
    print(f"  {name:6s}: 1500K {a[:,0].mean():6.1f} +/- {a[:,0].std():4.1f} | 1800K {a[:,1].mean():6.1f} +/- {a[:,1].std():4.1f} | strained {a[:,2].mean():6.1f} +/- {a[:,2].std():4.1f} | E {a[:,3].mean():5.2f} meV/atom | cover {a[:,4].mean():.2f}")
fps0 = strategies[('fps', 0)]
print("FPS (seed 0) picks per temperature:", {T: int((Ts[fps0] == T).sum()) for T in (300, 900, 1500)})
# PCA map for the figure
U, S, Vt = np.linalg.svd(Z_ - Z_.mean(0), full_matrices=False)
P = (Z_ - Z_.mean(0)) @ Vt[:2].T; Pr = (Zr - Z_.mean(0)) @ Vt[:2].T; Pt = (Zt - Z_.mean(0)) @ Vt[:2].T
print(f"PCA: first two components explain {100 * (S[:2] ** 2).sum() / (S ** 2).sum():.1f} % of SOAP variance")
np.savez(os.path.join(C.DATA, '104_data_fig.npz'), acf300=acf[300], acf900=acf[900], acf1500=acf[1500], P=P, Pr=Pr, Pt=Pt, T=Ts,
         fps=np.array(fps0), rnd=np.array(strategies[('random', 0)]), blk=np.array(strategies[('block', 0)]))
json.dump(summ, open(os.path.join(C.DATA, '104_data_summary.json'), 'w'), indent=1)
print(f"CPU {time.process_time() - t0:.0f} s")
