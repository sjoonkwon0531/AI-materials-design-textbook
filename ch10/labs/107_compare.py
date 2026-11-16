"""Lab 10.7: merge the four uMLIP results (data/107_<model>.json) and compare with published DFT.

Reference (verified): Evarestov, Blokhin, Gryaznov, Kotomin, Maier, PRB 83, 134108 (2011),
plane-wave (VASP) PBE, 2x2x2 supercell, a = 3.944 A (their Table 3), frequencies in cm^-1
(their Table 4): G TO1 133i, TO2 146, t2u 226, TO3 508; R4+ 86i, R3+ 433, R1+ 798.
Tilt angle (their Table 6): PW PBE 4.9 deg; experiment 2.1 deg (4 K). Experimental a = 3.905 A.

Modes are assigned by degeneracy: at G the non-acoustic triplets in ascending order are
TO1, TO2, t2u, TO3 (the PBE order); at R the lowest triplet is R4+, the doublet R3+ and the
singlet R1+; at M the lowest mode is the in-phase tilt M3+.

Run:  python3 107_compare.py     (any interpreter with numpy)
"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data')
MODELS = ['mace', 'chgnet', 'sevennet', 'mattersim']
LAB = {'mace': 'MACE-MP-0 small', 'chgnet': 'CHGNet 0.3.0', 'sevennet': 'SevenNet-0', 'mattersim': 'MatterSim 1M'}
PBE = {'G_TO1': -133, 'G_TO2': 146, 'G_t2u': 226, 'G_TO3': 508, 'R4+': -86, 'R3+': 433, 'R1+': 798}


def groups(f, tol=1.5):
    f = sorted(f); out = [[f[0]]]
    for x in f[1:]:
        (out[-1].append(x) if abs(x - out[-1][-1]) < tol else out.append([x]))
    return [(float(np.mean(g)), len(g)) for g in out]


def assign(fr):
    gG = [g for g in groups(fr['G']) if not (abs(g[0]) < 3 and g[1] == 3)]
    trip = [g[0] for g in gG if g[1] == 3]
    gR = groups(fr['R'])
    r = dict(G_TO1=trip[0], G_TO2=trip[1], G_t2u=trip[2], G_TO3=trip[3],
             **{'R4+': [g for g in gR if g[1] == 3][0][0], 'R3+': [g for g in gR if g[1] == 2][-1][0],
                'R1+': [g for g in gR if g[1] == 1][-1][0]},
             M3=min(fr['M']), X_min=min(fr['X']))
    return r


def s(x):
    return f'{abs(x):.0f}i' if x < 0 else f'{x:.0f}'


D = {m: json.load(open(os.path.join(DATA, f'107_{m}.json'))) for m in MODELS}
print('=== lattice constant, bulk modulus, cost ===')
for m in MODELS:
    d = D[m]
    print(f'  {LAB[m]:16s} a0 = {d["eos"]["a0"]:.4f} A ({100*(d["eos"]["a0"]/3.905-1):+.2f}% vs exp 3.905), B = {d["eos"]["B"]:.0f} GPa,'
          f' params {d["info"]["n_params"]:>9,}, CPU/40-atom call {1000*d["t_call"]:4.0f} ms, phonon CPU {d["ph"]["own"]["cpu"]:.1f} s')
a0s = [D[m]['eos']['a0'] for m in MODELS]
print(f'  spread of a0: {min(a0s):.4f}-{max(a0s):.4f} A (PBE/VASP 3.944)')

for tag, title in [('pbe', 'at the PBE lattice constant a = 3.944 A (same structure as the DFT column)'),
                   ('own', "at each model's own a0"), ('exp', 'at the experimental a = 3.905 A')]:
    print(f'\n=== frequencies (cm-1) {title} ===')
    keys = ['G_TO1', 'G_TO2', 'G_t2u', 'G_TO3', 'R4+', 'R3+', 'R1+', 'M3', 'X_min']
    print('  ' + f'{"":16s}' + ''.join(f'{k:>8s}' for k in keys))
    if tag == 'pbe':
        print('  ' + f'{"PBE (Evarestov)":16s}' + ''.join(f'{s(PBE[k]) if k in PBE else "-":>8s}' for k in keys))
    A = {}
    for m in MODELS:
        A[m] = assign(D[m]['ph'][tag]['freqs'])
        print('  ' + f'{LAB[m]:16s}' + ''.join(f'{s(A[m][k]):>8s}' for k in keys))
    if tag == 'pbe':
        print('  stable-mode ratio w(MLIP)/w(PBE) for G_TO2, G_t2u, G_TO3, R3+, R1+ and its mean; omega^2 ratio of soft modes:')
        for m in MODELS:
            rr = [A[m][k] / PBE[k] for k in ['G_TO2', 'G_t2u', 'G_TO3', 'R3+', 'R1+']]
            s2 = [(A[m][k] / PBE[k]) ** 2 * (np.sign(A[m][k]) * np.sign(PBE[k])) for k in ['G_TO1', 'R4+']]
            print(f'    {LAB[m]:16s} ' + ' '.join(f'{x:.3f}' for x in rr) + f'  mean {np.mean(rr):.3f} |'
                  f' w^2 ratio G_TO1 {s2[0]:.2f}, R4+ {s2[1]:.2f}')

print('\n=== a0a0c- tilt (2x2x2, I4/mcm relaxation of cell and positions) ===')
print('  PBE/VASP (Evarestov): 4.9 deg; experiment 2.1 deg (4 K)')
for m in MODELS:
    t = D[m]['tilt']; i = int(np.argmin(t['prof']))
    print(f'  {LAB[m]:16s} relaxed tilt {t["phi_rel"]:.2f} deg, well depth {t["dE_rel"]:.2f} meV/f.u., c/a {t["cell"][2]/t["cell"][0]:.4f};'
          f' rigid-tilt minimum {t["prof"][i]:.2f} meV/f.u. at {t["phis"][i]:.1f} deg, E(10 deg) {t["prof"][-1]:+.1f}')
ph = [D[m]['tilt']['phi_rel'] for m in MODELS]; dd = [D[m]['tilt']['dE_rel'] for m in MODELS]
print(f'  spread: tilt {min(ph):.2f}-{max(ph):.2f} deg, depth {min(dd):.2f} to {max(dd):.2f} meV/f.u.')

print('\n=== cost of the whole lattice-dynamics workflow per model (force calls) ===')
for m in MODELS:
    d = D[m]
    n = 9 + 3 * d['ph']['own']['ndisp'] + 1 + len(d['tilt']['phis']) + d['tilt']['steps'] + 1 + 5
    print(f'  {LAB[m]:16s} {n} calls (9 EOS of 5 atoms, {3*d["ph"]["own"]["ndisp"]} phonon supercells, '
          f'{len(d["tilt"]["phis"])+1} tilt points, {d["tilt"]["steps"]+1} relaxation steps, 5 timing), total CPU {d["cpu_total"]:.0f} s')

print('\n=== two routes to the R4+ curvature: frozen tilt profile E(u) versus phonopy forces ===')
print('  fit E = c2 u^2 + c4 u^4 per f.u. for tilt <= 6 deg (u = equatorial O displacement, two O per f.u. move),')
print('  then omega^2 = c2 / m_O ;  1 sqrt(eV/(A^2 amu)) = 521.47 cm-1')
for m in MODELS:
    d = D[m]; t = d['tilt']; a = d['eos']['a0']
    phi = np.array(t['phis']); E = np.array(t['prof']) / 1000; u = a / 2 * np.tan(np.radians(phi)); sel = phi <= 6
    c2, c4 = np.linalg.lstsq(np.vstack([u[sel] ** 2, u[sel] ** 4]).T, E[sel], rcond=None)[0]
    w = np.sign(c2) * np.sqrt(abs(c2) / 15.999) * 521.47
    print(f'  {LAB[m]:16s} c2 = {c2:+.3f} eV/A^2, c4 = {c4:.2f} eV/A^4 -> {s(w):>4s} cm-1 ; phonopy R4+ (0.01 A) {s(min(d["ph"]["own"]["freqs"]["R"])):>4s} cm-1')
