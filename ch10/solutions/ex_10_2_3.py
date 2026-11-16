"""Ex 10.2.3: stress weight sweep, EqNet rc=3.0, 40 training cells, E+F(+S), same budget as Lab 10.2 (B).
Run with $MLIP_PY_A (MACE/CHGNet environment). wS=100 reproduces the E+F+S row of Lab 10.2; wS=0 the E+F row."""
import os, sys, json, importlib.util
import numpy as np, torch
torch.set_num_threads(1)
LABS = '../../ch10/labs'
sys.argv = [sys.argv[0]] + sys.argv[1:]
spec = importlib.util.spec_from_file_location('s102', os.path.join(LABS, '102_stress.py'))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
T, M = S.T, S.M
tr, te = T.load_data(); e0 = (tr['E'] / 8).mean().item()
ws_list = [float(x) for x in (sys.argv[1:] or ['0', '10', '100', '1000'])]
for wS in ws_list:
    m = T.build('eq', 3.0, e0=e0, seed=0)
    cpu = T.train(m, tr, np.arange(40), steps=1500, bs=8, seed=0, wE=10.0, wF=1.0, wS=wS)
    ev = T.evaluate(m, te)
    g = M.Graph(te['R'].numpy(), te['cell'].numpy(), 3.0)
    Ep = M.predict(m, te['R'].reshape(-1, 3), te['cell'], g)['E']; d = (Ep - te['E']) / 8
    sm = S.stress_mae(m, te)
    # pressure error separately (trace part) and deviatoric part
    out = M.predict(m, te['R'].reshape(-1, 3), te['cell'], g, stress=True)
    P_ml = -out['S'].diagonal(dim1=1, dim2=2).sum(-1) / 3 * S.GPA; P_ref = -te['S'].diagonal(dim1=1, dim2=2).sum(-1) / 3 * S.GPA
    print(f"wS={wS:6.0f} | E MAE {ev['E_mae']:6.1f} meV/atom (offset {d.mean().item()*1e3:+6.1f}) | F MAE {ev['F_mae']:5.1f} meV/A"
          f" | S MAE {sm:5.2f} GPa | P MAE {(P_ml-P_ref).abs().mean().item():5.2f} GPa | cpu {cpu:4.0f} s", flush=True)
