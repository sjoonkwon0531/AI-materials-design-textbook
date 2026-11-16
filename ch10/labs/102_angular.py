"""Lab 10.2 (102_angular): when do the l = 1 channels pay off?  Invariant vs equivariant students on
periodic Si, with a short cutoff (first neighbour shell only) and a longer cutoff (second shell included).

Oracle: MACE-MP-0 small (stands in for DFT).  Data: ../data/102_si8.npz (102_data.py).
Students: InvNet (scalars only, distances only) and EqNet (scalars + vectors), both 2 layers, 32 channels.
Same training budget for every run: 400 training cells, 2500 Adam steps, batch 16, lr 3e-3 cosine, wE = 10, wF = 1, seed 0.
(A first pass with 1000 steps x batch 8 was under-trained: 400 cells did worse than 50 cells.)
Run with /tmp/claude-0/v10/bin/python (one thread).
"""
import os, json, time, importlib.util
import numpy as np
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t102', os.path.join(HERE, '102_train.py'))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)


def main():
    tr, te = T.load_data()
    e0 = (tr['E'] / 8).mean().item()
    # neighbour counts for the two cutoffs (mean over test cells)
    for rc in [3.0, 4.2]:
        g = T.M.Graph(te['R'].numpy(), te['cell'].numpy(), rc)
        print(f"rc = {rc:.1f} A: mean neighbours per atom = {len(g.i) / (g.nb * g.n_at):.2f}")
    zeroF = te['F'].abs().mean().item() * 1e3
    print(f"zero-force baseline on test: F MAE = {zeroF:.0f} meV/A")
    res = {}
    for rc in [3.0, 4.2]:
        for N in [400]:
            for kind in ['inv', 'eq']:
                m = T.build(kind, rc, e0=e0, seed=0)
                cpu = T.train(m, tr, np.arange(N), steps=2500, bs=16, wE=10.0, wF=1.0, seed=0)
                ev = T.evaluate(m, te)
                res[f'{kind}_rc{rc}_N{N}'] = dict(ev, cpu=cpu, params=T.M.count_params(m))
                print(f"{kind:3s} rc={rc:.1f} N={N:3d} params={T.M.count_params(m):6d} | "
                      f"E MAE {ev['E_mae']:6.1f} meV/atom  F MAE {ev['F_mae']:5.0f} meV/A | cpu {cpu:5.1f} s", flush=True)
    json.dump(res, open(os.path.join(T.DATA, '102_angular.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
