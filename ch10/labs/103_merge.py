"""Lab 10.3: merge data/103_<model>.json from the two interpreters and print the comparison tables."""
import os, json
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS = ['mace', 'chgnet', 'sevennet', 'mattersim']
R = {m: json.load(open(os.path.join(DATA, f'103_{m}.json'))) for m in MODELS}
ELEMS = ['Al', 'Cu', 'Ag', 'Au', 'Si', 'Ge', 'Mo', 'W']
short = {'mace': 'MACE', 'chgnet': 'CHGNet', 'sevennet': '7net-0', 'mattersim': 'MatSim'}

print('=== A. models as loaded (parameter count, dtype) ===')
for m in MODELS:
    i = R[m]['info']; print(f"  {i['label']:22s} params = {i['n_params']:>9,d}  dtype = {i['dtype']}")

print('\n=== B. EOS vs PBE (WIEN2k) and experiment: lattice-volume error (%) and B0 (GPa) ===')
print('  elem |' + ''.join(f'{short[m]:>9s}' for m in MODELS) + '|   PBE-exp  (dV0 % relative to PBE)')
dv = {m: [] for m in MODELS}; db = {m: [] for m in MODELS}
for el in ELEMS:
    e0 = R['mace']['eos'][el]
    row = []
    for m in MODELS:
        e = R[m]['eos'][el]; x = 100 * (e['V0'] / e['V0_pbe'] - 1); dv[m].append(x); row.append(x)
        db[m].append(e['B0'] / e['B0_pbe'])
    print(f'  {el:4s} |' + ''.join(f'{x:+9.2f}' for x in row) + f"|  {100*(e0['V0_exp']/e0['V0_pbe']-1):+7.2f}")
print('  MAE  |' + ''.join(f'{np.mean(np.abs(dv[m])):9.2f}' for m in MODELS) + '|  ' +
      f"{np.mean([abs(100*(R['mace']['eos'][el]['V0_exp']/R['mace']['eos'][el]['V0_pbe']-1)) for el in ELEMS]):7.2f}")
print('\n  B0 (GPa)')
print('  elem |' + ''.join(f'{short[m]:>9s}' for m in MODELS) + '|      PBE      exp')
for k, el in enumerate(ELEMS):
    e0 = R['mace']['eos'][el]
    print(f'  {el:4s} |' + ''.join(f"{R[m]['eos'][el]['B0']:9.1f}" for m in MODELS) + f"| {e0['B0_pbe']:8.1f} {e0['B0_exp']:8.1f}")
print('  mean B0(model)/B0(PBE) |' + ''.join(f'{np.mean(db[m]):9.3f}' for m in MODELS))
print('  elements with B0(model) < B0(PBE):', {short[m]: int(np.sum(np.array(db[m]) < 1)) for m in MODELS}, 'of', len(ELEMS))

print('\n=== C. elastic constants (GPa), relaxed ions for Si; strain 0.5 % / 1.0 % ===')
for sym, ref in (('Si', '(exp: C11 165.7, C12 63.9, C44 79.5)'), ('Cu', '')):
    print(f'  {sym} {ref}')
    for m in MODELS:
        a, b = R[m]['elastic'][f'{sym}_0.005'], R[m]['elastic'][f'{sym}_0.01']
        print(f"   {short[m]:7s} C11 {a['C11']:6.1f}/{b['C11']:6.1f}  C12 {a['C12']:6.1f}/{b['C12']:6.1f}  "
              f"C44 {a['C44']:6.1f}/{b['C44']:6.1f}   (C11+2C12)/3 = {a['B_from_C']:6.1f}  B_EOS = {a['B_eos']:6.1f}")

print('\n=== D. stress = dE/dstrain? (Si, at the model EOS a0) and B from 0.5 % vs 1 % strain ===')
for m in MODELS:
    s = R[m]['stresscheck']['Si']
    print(f"   {short[m]:7s} sigma_xx: calc {s['stress_xx_calc_GPa']:+.3f}, FD {s['stress_xx_fd_GPa']:+.3f} GPa | "
          f"B from E'': {s['B_E_0.005']:6.1f} / {s['B_E_0.01']:6.1f} GPa")

print('\n=== E. CPU time per energy+force call (1 thread, median of 3) ===')
for m in MODELS:
    t = R[m]['timing']
    print(f"   {short[m]:7s} " + '  '.join(f"N={n}: {v['cpu_s']:6.3f} s ({v['us_per_atom']/1e3:5.2f} ms/atom)" for n, v in t.items()))

print('\n=== F. rotation equivariance (NaCl, 64 atoms, rattled 0.1 A, 5 rotations) ===')
for m in MODELS:
    q = R[m]['equiv']
    print(f"   {short[m]:7s} ({q['dtype']}) max|dE|/N = {q['dE_per_atom']:.1e} eV  max|dF| = {q['dF_max']:.1e} eV/A  "
          f"max|dsigma| = {q['dS_max_GPa']:.1e} GPa")

print('\n=== G. NVE, Si 64 atoms, MB 1200 K start, dt 2 fs, 400 steps ===')
for m in MODELS:
    q = R[m]['nve']
    print(f"   {short[m]:7s} <T> = {q['T_mean']:5.0f} K  drift = {q['drift_meV_atom_ps']:+7.3f} meV/atom/ps  "
          f"std(E) = {q['fluct_meV_atom']:.3f} meV/atom  CPU {q['cpu_s']:.0f} s")

print('\n=== H. OOD probes ===')
for m in MODELS:
    o = R[m]['ood']
    print(f"   {short[m]:7s} Si2: r_min {o['Si2']['r_min']:.1f} A, E_min {o['Si2']['E_min']:+.2f} eV, E(1.0 A) {o['Si2']['E_at_1A']:+.2f} eV, "
          f"monotone {o['Si2']['monotone_inside']} | Cu2: r_min {o['Cu2']['r_min']:.1f}, E(1.0 A) {o['Cu2']['E_at_1A']:+.2f} eV, "
          f"monotone {o['Cu2']['monotone_inside']}")
    print(f"           Si compressed P (GPa) at V/V0 = 1.0..0.6: " + ', '.join(f'{p:.1f}' for p in o['Si_compress']['P_GPa']))
