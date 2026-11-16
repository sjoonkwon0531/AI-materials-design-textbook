"""Ex 10.3.3: V0 errors of the four models relative to PBE and to experiment (Lab 10.5 Part B data)."""
import json, numpy as np
D = '../ch10/data/'
names = {'mace': 'MACE', 'chgnet': 'CHGNet', 'sevennet': '7net-0', 'mattersim': 'MatSim'}
els = ['Al', 'Cu', 'Ag', 'Au', 'Si', 'Ge', 'Mo', 'W']
for f, n in names.items():
    e = json.load(open(D + f'103_{f}.json'))['eos']
    vp = np.array([100 * (e[x]['V0'] / e[x]['V0_pbe'] - 1) for x in els])
    ve = np.array([100 * (e[x]['V0'] / e[x]['V0_exp'] - 1) for x in els])
    print(f"{n:7s}: MAPE vs PBE {np.abs(vp).mean():.2f} %, vs exp {np.abs(ve).mean():.2f} %; signed vs exp " + ' '.join(f'{x:+.2f}' for x in ve))
pe = np.array([100 * (e[x]['V0_pbe'] / e[x]['V0_exp'] - 1) for x in els])
pe2 = np.array([100 * (e[x]['V0_exp'] / e[x]['V0_pbe'] - 1) for x in els])
print(f"PBE vs exp (relative to exp): MAPE {np.abs(pe).mean():.2f} %; per element " + ' '.join(f'{x:+.2f}' for x in pe))
print(f"(table column, relative to PBE): MAPE {np.abs(pe2).mean():.2f} %")
