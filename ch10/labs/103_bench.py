"""Lab 10.3 (core benchmark): four universal MLIPs on one small, well-defined test suite.

Usage:  python 103_bench.py <model> [part ...]
  model in {mace, chgnet, sevennet, mattersim}; the script re-executes itself under the venv
  that has the model (see 103_common.py).  Parts: info eos elastic timing equiv nve ood (default all).
Results are merged into data/103_<model>.json; 103_merge.py prints the comparison tables.

Reference data (stated in the text):
  EOS: Delta-test set bundled with ASE (ase.collections.dcdft), i.e. the WIEN2k all-electron PBE
       equation of state and zero-point-corrected experimental V0, B0 compiled by
       Lejaeghere et al., Crit. Rev. Solid State Mater. Sci. 39, 1 (2014).
  Elastic: Si experiment C11=165.7, C12=63.9, C44=79.5 GPa (Decremps et al., arXiv:1007.2730).
All runs: torch threads = 1, CPU time from time.process_time().
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('103_common')

model = sys.argv[1]
C.reexec_for(model)
parts = sys.argv[2:] or ['info', 'eos', 'elastic', 'timing', 'equiv', 'nve', 'ood']

import numpy as np
from ase.collections import dcdft
from ase.eos import EquationOfState
from ase.build import bulk
from ase import Atoms, units
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

calc, net, info = C.make_calc(model)
GPa = units.GPa
ELEMS = ['Al', 'Cu', 'Ag', 'Au', 'Si', 'Ge', 'Mo', 'W']


def ep(atoms):
    atoms.calc = calc
    return atoms.get_potential_energy()


if 'info' in parts:
    print(f"[{model}] {info}")
    C.save(model, 'info', info)

# ---------------------------------------------------------------- EOS (Delta protocol, 7 volumes, 94-106 %)
if 'eos' in parts:
    t0 = time.process_time(); out = {}
    for el in ELEMS:
        ref = dcdft[el]; d = dcdft.data[el]; nat = len(ref)
        s = d['wien2k_volume'] * nat / ref.get_volume()
        vols, ens = [], []
        for f in np.linspace(0.94, 1.06, 7):
            a = ref.copy(); a.set_cell(ref.cell * (s * f) ** (1 / 3), scale_atoms=True)
            vols.append(a.get_volume() / nat); ens.append(ep(a) / nat)
        v0, e0, B = EquationOfState(vols, ens, eos='birchmurnaghan').fit()
        out[el] = dict(V0=float(v0), B0=float(B / GPa), V0_pbe=d['wien2k_volume'], B0_pbe=d['wien2k_B'],
                       V0_exp=d.get('exp_volume'), B0_exp=d.get('exp_B'))
        print(f"[{model}] EOS {el:2s}: V0 = {v0:7.3f} A^3/atom (PBE {d['wien2k_volume']:7.3f}, exp {d['exp_volume']:7.3f})"
              f"  B0 = {B/GPa:6.1f} GPa (PBE {d['wien2k_B']:6.1f}, exp {d['exp_B']:6.1f})")
    out['cpu_s'] = round(time.process_time() - t0, 1)
    C.save(model, 'eos', out)

# ---------------------------------------------------------------- elastic constants (stress-strain, relaxed ions)
def cubic_elastic(sym, a0, crystal, relax_ions, eps):
    base = bulk(sym, crystal, a=a0, cubic=True)

    def stress(strain):
        a = base.copy()
        a.set_cell(np.dot(base.cell, np.eye(3) + strain), scale_atoms=True)
        a.calc = calc
        if relax_ions:
            BFGS(a, logfile=None).run(fmax=1e-3, steps=200)
        return a.get_stress(voigt=True) / GPa    # ASE: stress = (1/V) dE/d(strain), tensile positive
    # uniaxial e_xx -> C11 = dsigma_xx/de, C12 = dsigma_yy/de
    sp = stress(np.diag([eps, 0, 0])); sm = stress(np.diag([-eps, 0, 0]))
    C11 = (sp[0] - sm[0]) / (2 * eps); C12 = 0.5 * ((sp[1] - sm[1]) + (sp[2] - sm[2])) / (2 * eps)
    # shear gamma_yz = 2 e_yz
    e = np.zeros((3, 3)); e[1, 2] = e[2, 1] = eps / 2
    sp = stress(e); sm = stress(-e)
    C44 = (sp[3] - sm[3]) / (2 * eps)
    return dict(C11=float(C11), C12=float(C12), C44=float(C44))


if 'elastic' in parts:
    t0 = time.process_time(); out = {}
    d = json.load(open(os.path.join(C.DATA, f'103_{model}.json')))['eos']
    for sym, crystal, nat in (('Si', 'diamond', 8), ('Cu', 'fcc', 4)):
        a0 = (d[sym]['V0'] * nat) ** (1 / 3)
        for eps in (0.005, 0.01):                         # two strain amplitudes: is the result stable?
            r = cubic_elastic(sym, a0, crystal, relax_ions=(crystal == 'diamond'), eps=eps)
            r['B_from_C'] = (r['C11'] + 2 * r['C12']) / 3; r['a0'] = a0; r['B_eos'] = d[sym]['B0']
            out[f'{sym}_{eps}'] = r
            print(f"[{model}] elastic {sym} (strain {100*eps:.1f}%): a0 = {a0:.4f} A, C11 = {r['C11']:6.1f}, C12 = {r['C12']:6.1f}, "
                  f"C44 = {r['C44']:6.1f} GPa; (C11+2C12)/3 = {r['B_from_C']:6.1f} vs B_EOS = {r['B_eos']:6.1f}")
    out['cpu_s'] = round(time.process_time() - t0, 1)
    C.save(model, 'elastic', out)

# ---------------------------------------------------------------- timing per atom (Si supercells)
if 'timing' in parts:
    out = {}
    for n in (1, 2, 3):                                   # 8, 64, 216 atoms (512 exceeded the shared memory limit)
        a = bulk('Si', 'diamond', a=5.47, cubic=True) * (n, n, n)
        a.rattle(0.02, seed=1); a.calc = calc
        a.get_forces()                                     # warm-up (graph build, JIT)
        ts = []
        for k in range(3):
            a.positions += 1e-4                            # invalidate cache
            t0 = time.process_time(); a.get_potential_energy(); a.get_forces(); ts.append(time.process_time() - t0)
        t = float(np.median(ts)); out[len(a)] = dict(cpu_s=t, us_per_atom=1e6 * t / len(a))
        print(f"[{model}] timing N={len(a):4d}: {t:7.3f} s CPU per energy+force call, {1e3*t/len(a):6.2f} ms/atom")
    C.save(model, 'timing', out)

# ---------------------------------------------------------------- rotation equivariance
if 'equiv' in parts:
    from scipy.spatial.transform import Rotation
    a = bulk('NaCl', 'rocksalt', a=5.64, cubic=True) * (2, 2, 2)   # 64 atoms, two elements
    a.rattle(0.1, seed=3); a.calc = calc
    E0 = a.get_potential_energy(); F0 = a.get_forces().copy(); S0 = a.get_stress(voigt=False).copy()
    dE, dF, dS = [], [], []
    for k in range(5):
        R = Rotation.random(random_state=10 + k).as_matrix()
        b = a.copy(); b.set_cell(a.cell @ R.T, scale_atoms=False); b.positions = a.positions @ R.T
        b.calc = calc
        dE.append(abs(b.get_potential_energy() - E0) / len(a))
        dF.append(np.abs(b.get_forces() - F0 @ R.T).max())
        dS.append(np.abs(b.get_stress(voigt=False) - R @ S0 @ R.T).max() / GPa)
    out = dict(dE_per_atom=float(max(dE)), dF_max=float(max(dF)), dS_max_GPa=float(max(dS)),
               F_rms=float(np.sqrt((F0 ** 2).mean())), dtype=info['dtype'])
    print(f"[{model}] equivariance (NaCl 64 at, 5 random rotations): max|dE|/N = {out['dE_per_atom']:.2e} eV, "
          f"max|F(Rx)-RF(x)| = {out['dF_max']:.2e} eV/A (F_rms {out['F_rms']:.2f}), max|dsigma| = {out['dS_max_GPa']:.2e} GPa")
    C.save(model, 'equiv', out)

# ---------------------------------------------------------------- NVE energy conservation
if 'nve' in parts:
    a = bulk('Si', 'diamond', a=5.47, cubic=True) * (2, 2, 2)      # 64 atoms
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=1200, rng=np.random.default_rng(7))
    Stationary(a)
    dt_fs, nsteps = 2.0, 400
    dyn = VelocityVerlet(a, dt_fs * units.fs)
    Et, Tt = [], []
    def rec():
        Et.append(a.get_total_energy() / len(a)); Tt.append(a.get_temperature())
    dyn.attach(rec, interval=1)
    t0 = time.process_time(); dyn.run(nsteps); cpu = time.process_time() - t0
    Et = np.array(Et); t_ps = np.arange(len(Et)) * dt_fs / 1000
    half = len(Et) // 4                                       # discard first 0.2 ps (equipartition)
    slope = np.polyfit(t_ps[half:], Et[half:], 1)[0]
    out = dict(dt_fs=dt_fs, nsteps=nsteps, drift_meV_atom_ps=float(1e3 * slope),
               fluct_meV_atom=float(1e3 * Et[half:].std()), T_mean=float(np.mean(Tt[half:])),
               cpu_s=round(cpu, 1), E=[float(x) for x in (Et - Et[0]) * 1e3], T=[float(x) for x in Tt])
    print(f"[{model}] NVE Si64, dt {dt_fs} fs, {nsteps} steps: <T> = {out['T_mean']:.0f} K, drift = "
          f"{out['drift_meV_atom_ps']:+.3f} meV/atom/ps, std(E_tot) = {out['fluct_meV_atom']:.3f} meV/atom, CPU {cpu:.0f} s")
    C.save(model, 'nve', out)

# ---------------------------------------------------------------- OOD probes: dimer at short range, strong compression
if 'ood' in parts:
    out = {}
    rs = np.round(np.arange(0.6, 4.01, 0.1), 2)
    for el in ('Si', 'Cu'):
        Es = []
        for r in rs:
            d = Atoms(el + '2', positions=[[0, 0, 0], [r, 0, 0]], cell=[18, 18, 18], pbc=True)
            Es.append(ep(d))
        Es = np.array(Es); Es -= Es[-1]
        i = int(np.argmin(Es)); dE = np.diff(Es[:i + 1])
        out[el + '2'] = dict(r=rs.tolist(), E=Es.tolist(), r_min=float(rs[i]), E_min=float(Es[i]),
                              monotone_inside=bool(np.all(dE < 0)), E_at_1A=float(Es[np.argmin(abs(rs - 1.0))]))
        print(f"[{model}] dimer {el}2: r_min = {rs[i]:.1f} A, E_min = {Es[i]:.2f} eV, E(1.0 A) = {out[el+'2']['E_at_1A']:+.2f} eV, "
              f"repulsive wall monotone below r_min: {out[el+'2']['monotone_inside']}")
    # compressed diamond Si: V/V0 from 1.0 down to 0.6 (pressure check)
    ref = dcdft['Si']; d = dcdft.data['Si']; nat = len(ref)
    s = d['wien2k_volume'] * nat / ref.get_volume(); P = []
    fr = [1.0, 0.9, 0.8, 0.7, 0.6]
    for f in fr:
        a = ref.copy(); a.set_cell(ref.cell * (s * f) ** (1 / 3), scale_atoms=True); a.calc = calc
        P.append(float(-np.trace(a.get_stress(voigt=False)) / 3 / GPa))
    out['Si_compress'] = dict(V_over_V0=fr, P_GPa=P)
    print(f"[{model}] Si diamond compressed: " + ", ".join(f"V/V0={f:.1f}: P={p:6.1f} GPa" for f, p in zip(fr, P)))
    C.save(model, 'ood', out)
