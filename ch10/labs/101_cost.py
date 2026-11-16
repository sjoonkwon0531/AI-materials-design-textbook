"""Lab 10.1 (101_cost): how the cost of one energy+force call grows with the number of atoms N.

Three engines on the same CPU, one thread each, CPU time (process_time):
  (1) Kohn-Sham DFT, PySCF periodic RKS, PBE, gth-szv basis, gth-pade pseudopotential, Gamma point,
      diamond Si cells with N = 2 and 8 atoms (larger cells were not run in the shared environment) (one full SCF each).   -> run by:  python3 101_cost.py dft
  (2) MACE-MP-0 small (universal MLIP), Si supercells N = 8 ... 1000.  -> run by:  v10/bin/python 101_cost.py mlip
  (3) ASE EMT (classical embedded-atom-like potential), Cu fcc N = 108 ... 4000, as the classical floor.
Results are written to ../data/101_cost.json; the figure script reads that file.
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
OUT = os.path.join(DATA, '101_cost.json')
os.environ.setdefault('OMP_NUM_THREADS', '1')


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(d):
    json.dump(d, open(OUT, 'w'), indent=1)


def run_dft(Ns=(2, 8)):
    from pyscf.pbc import gto, dft
    from ase.build import bulk
    res = load(); res.setdefault('dft', {})
    for N in Ns:
        if str(N) in res['dft']:
            print(f"DFT  N={N:3d}  (cached)  E={res['dft'][str(N)]['E']:.6f} Ha  cpu={res['dft'][str(N)]['cpu']:.1f} s")
            continue
        if N == 2:
            at = bulk('Si', 'diamond', a=5.43)
        else:
            at = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((N // 8, 1, 1))
        cell = gto.Cell()
        cell.atom = [(s, p) for s, p in zip(at.get_chemical_symbols(), at.positions)]
        cell.a = at.cell[:]; cell.basis = 'gth-szv'; cell.pseudo = 'gth-pade'; cell.verbose = 0
        cell.build()
        t = time.process_time()
        mf = dft.RKS(cell); mf.xc = 'pbe'; e = mf.kernel()
        cpu = time.process_time() - t
        res['dft'][str(N)] = {'E': float(e), 'cpu': cpu, 'nao': int(cell.nao_nr()), 'conv': bool(mf.converged)}
        print(f"DFT  N={N:3d}  nao={cell.nao_nr():4d}  E={e:.6f} Ha  cpu={cpu:.1f} s", flush=True)
        save(res)


def run_mlip():
    import torch
    torch.set_num_threads(1)
    from ase.build import bulk
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=os.path.join(DATA, 'mace_mp0_small.model'),
                          device='cpu', default_dtype='float64')
    res = load(); res["mlip"] = {}
    rng = __import__('numpy').random.default_rng(0)
    for rep in [1, 2, 3, 4, 5]:
        at = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((rep, rep, rep))
        at.positions += rng.normal(0, 0.03, at.positions.shape)
        at.calc = calc
        at.get_forces()                          # warm-up (graph build, JIT paths)
        nrep = 3 if len(at) <= 512 else 1
        t = time.process_time()
        for _ in range(nrep):
            at.calc.results = {}
            calc.calculate(at, ['energy', 'forces'])
        cpu = (time.process_time() - t) / nrep
        res['mlip'][str(len(at))] = cpu
        print(f"MACE N={len(at):5d}  cpu/call={cpu:.3f} s  per atom={1e3*cpu/len(at):.2f} ms", flush=True)
        save(res)
    # classical floor
    from ase.calculators.emt import EMT
    res['emt'] = {}
    for rep in [3, 5, 8, 10]:
        at = bulk('Cu', 'fcc', a=3.61, cubic=True).repeat((rep, rep, rep))
        at.calc = EMT(); at.get_forces()
        t = time.process_time()
        for _ in range(3):
            at.calc.results = {}; at.get_forces()
        cpu = (time.process_time() - t) / 3
        res['emt'][str(len(at))] = cpu
        print(f"EMT  N={len(at):5d}  cpu/call={cpu:.4f} s  per atom={1e6*cpu/len(at):.1f} us", flush=True)
    save(res)


def summary():
    import numpy as np
    r = load()
    Nd = np.array(sorted(int(k) for k in r['dft'])); td = np.array([r['dft'][str(n)]['cpu'] for n in Nd])
    Nm = np.array(sorted(int(k) for k in r['mlip'])); tm = np.array([r['mlip'][str(n)] for n in Nm])
    pd = np.polyfit(np.log(Nd), np.log(td), 1)[0]
    pm = np.polyfit(np.log(Nm[2:]), np.log(tm[2:]), 1)[0]
    print(f"measured slope d ln t / d ln N:  DFT (N={Nd.min()}-{Nd.max()}) {pd:.2f}   MACE (N>={Nm[2]}) {pm:.2f}")
    # two DFT models anchored at the largest measured DFT cell: the measured small-N slope (optimistic)
    # and the asymptotic cubic law of diagonalisation (Ch.4 sec 4.3.1)
    N0, t0 = Nd[-1], td[-1]
    for N in [64, 216, 1000]:
        tc = t0 * (N / N0) ** 3; tp = t0 * (N / N0) ** pd
        tml = np.interp(N, Nm, tm)
        print(f"N={N:5d}: DFT model  slope {pd:.2f}: {tp:9.3g} s   cubic: {tc:9.3g} s   | MACE measured {tml:7.3f} s"
              f"   ratio {tp/tml:8.3g} .. {tc/tml:8.3g}")
    # per atom-step costs and what '1 nm -> 1 um' would take on this core
    c_ml = tm[-1] / Nm[-1]; c_emt = np.mean([r['emt'][k] / int(k) for k in r['emt']])
    print(f"per atom-step: MACE {1e3*c_ml:.1f} ms, EMT {1e6*c_emt:.0f} us, MACE/EMT = {c_ml/c_emt:.0f}")
    n_si = 8 / 5.43 ** 3                                   # atoms per A^3 in diamond Si
    for L_nm in [1, 10, 100, 1000]:
        N = n_si * (10 * L_nm) ** 3
        t_step = N * c_ml
        print(f"Si cube {L_nm:5d} nm: N = {N:9.2e} atoms; one MACE step {t_step:9.2e} s; "
              f"1 ns (1e6 steps of 1 fs) {t_step*1e6/3.15e7:9.2e} core-years")
    # break-even number of MD steps (Estimate Box): labels on 8-atom cells, production on N atoms
    n_lab, C_train = 400, 100.0
    for N in [64, 216]:
        c_dft_p = t0 * (N / N0) ** pd; c_dft_c = t0 * (N / N0) ** 3; c_m = np.interp(N, Nm, tm)
        build = n_lab * t0 + C_train
        print(f"break-even (N={N}, {n_lab} labels of 8 atoms, C_train={C_train:.0f} s): build cost {build:.3g} s; "
              f"S* = {build/(c_dft_p-c_m):.3g} steps (slope model) .. {build/(c_dft_c-c_m):.3g} steps (cubic)")
    # ROI in DFT-call units: atom-steps obtained per DFT call
    S_md, N_md = 1_000_000, 64
    print(f"ROI: 1 ns (1e6 steps) of {N_md} atoms = {S_md*N_md:.1e} atom-steps; MLIP route {S_md*N_md/n_lab:.1e} atom-steps "
          f"per DFT call ({n_lab} calls); AIMD route {N_md} atom-steps per DFT call; ratio {S_md/n_lab:.0f}")
    r['summary'] = {'slope_dft': pd, 'slope_mace': pm, 'c_ml': c_ml, 'c_emt': c_emt}
    save(r)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    {'dft': run_dft, 'mlip': run_mlip, 'summary': summary}[what]()
