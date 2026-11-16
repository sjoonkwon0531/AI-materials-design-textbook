"""Ex 10.2.5: width-matched InvNet vs EqNet at rc = 3.0 A, Lab 10.4 budget (400 cells, 2500 steps, batch 16,
lr 3e-3 cosine, wE=10, wF=1). Usage: python ex_10_2_5.py KIND WIDTH SEED   (v10 interpreter, one thread)."""
import os, sys, importlib.util
import numpy as np, torch
torch.set_num_threads(1)
LABS = '../ch10/labs'
spec = importlib.util.spec_from_file_location('t102', os.path.join(LABS, '102_train.py'))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
kind, F, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tr, te = T.load_data(); e0 = (tr['E'] / 8).mean().item()
torch.manual_seed(seed)
net = (T.M.InvNet if kind == 'inv' else T.M.EqNet)(rc=3.0, F=F, T=2)
m = T.M.Potential(net, e0=e0)
cpu = T.train(m, tr, np.arange(400), steps=2500, bs=16, wE=10.0, wF=1.0, seed=seed)
ev = T.evaluate(m, te)
print(f"{kind} F={F} seed={seed} params={T.M.count_params(m)} | E MAE {ev['E_mae']:.1f} meV/atom  F MAE {ev['F_mae']:.1f} meV/A | cpu {cpu:.0f} s", flush=True)
