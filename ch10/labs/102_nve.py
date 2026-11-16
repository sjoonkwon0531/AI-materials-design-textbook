"""Lab 10.2 (102_nve): conservative forces vs a direct-force head, same backbone, same data.
 (A) Train two EqNet students (rc = 3.0 A, 2 layers, 32 channels) on 400 oracle-labelled 8-atom Si cells:
       cons   : E = sum eps_i, F = -dE/dR (autograd)
       direct : same backbone, energy head trained on E, separate vector head trained on F
     2500 Adam steps, batch 16, lr 3e-3, wE = 10, wF = 1, seed 0.
 (B) NVE velocity-Verlet MD of a 64-atom cell (2x2x2), start 600 K (Maxwell-Boltzmann, seed 0, zero momentum),
     5 ps at dt = 1.0 fs and at dt = 0.5 fs. Monitor E_model + K, and the oracle energy E_oracle + K every 250 fs.
 (C) Second derivatives: Hessian of the perfect 8-atom cell by double autograd vs finite differences of forces;
     frequencies of the 24 modes at the k-points folded into this cell (Gamma and X of the primitive cell),
     student vs oracle (oracle Hessian by finite differences).
Run with $MLIP_PY_A (MACE/CHGNet environment).
"""
import os, json, time, importlib.util, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t102', os.path.join(HERE, '102_train.py'))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
M = T.M
from ase.build import bulk
from ase import Atoms, units
from mace.calculators import MACECalculator

MSI = 28.0855
KB = 8.617333e-5
# ASE time unit: with masses in amu, energies in eV, lengths in A, one ASE time unit = units.fs**-1 fs
FS = units.fs


def md(model, at0, dt_fs, t_ps, oracle, seed=0, T0=600.0, every_fs=250.0):
    n = len(at0); m = MSI
    rng = np.random.default_rng(seed)
    v = rng.normal(0, np.sqrt(KB * T0 / m), (n, 3)); v -= v.mean(0)          # sqrt(eV/amu)
    v *= np.sqrt(KB * T0 * (3 * n - 3) / (m * (v ** 2).sum()))                 # exact initial T
    R = at0.positions.copy(); C = torch.as_tensor(at0.cell[:])[None]
    g = M.Graph(R[None], at0.cell[:][None], model.net.rc, skin=0.8)

    def ef(R):
        out = M.predict(model, torch.as_tensor(R), C, g)
        return out['E'].item(), out['F'][0].numpy()
    dt = dt_fs * FS
    E, F = ef(R)
    nsteps = int(round(t_ps * 1000 / dt_fs)); stride = int(round(every_fs / dt_fs))
    rec_t, rec_E, rec_T, orc_t, orc_E = [], [], [], [], []
    t0 = time.process_time()
    for s in range(nsteps + 1):
        if s % 10 == 0:
            K = 0.5 * m * (v ** 2).sum()
            rec_t.append(s * dt_fs / 1000); rec_E.append(E + K); rec_T.append(2 * K / ((3 * n - 3) * KB))
        if s % stride == 0:
            at = Atoms('Si%d' % n, positions=R, cell=at0.cell, pbc=True); at.calc = oracle
            orc_t.append(s * dt_fs / 1000); orc_E.append(at.get_potential_energy() + 0.5 * m * (v ** 2).sum())
        if s == nsteps:
            break
        v += 0.5 * dt * F / m
        R = R + dt * v
        E, F = ef(R)
        v += 0.5 * dt * F / m
        if not np.isfinite(E):
            break
    cpu = time.process_time() - t0
    rec_t, rec_E, orc_t, orc_E = map(np.array, (rec_t, rec_E, orc_t, orc_E))
    disp = np.abs(R - at0.positions).max()
    com = np.linalg.norm((R - at0.positions).mean(0))
    return dict(com=com, t=rec_t.tolist(), E=((rec_E - rec_E[0]) / n * 1e3).tolist(), T=rec_T,
                to=orc_t.tolist(), Eo=((orc_E - orc_E[0]) / n * 1e3).tolist(), cpu=cpu, disp=disp)


def slope(t, y):
    return np.polyfit(t, y, 1)[0]


def hessian_ad(model, R, C):
    g = M.Graph(R[None], C[None], model.net.rc, skin=0.3)
    Rt = torch.as_tensor(R).reshape(-1).clone().requires_grad_(True)
    out = M.predict(model, Rt.reshape(-1, 3), torch.as_tensor(C)[None], g, create_graph=True)
    F = out['F'].reshape(-1)
    H = torch.stack([-torch.autograd.grad(F[k], Rt, retain_graph=True)[0] for k in range(len(F))])
    return H.detach().numpy()


def hessian_fd(force_fn, R, h=1e-3):
    n = R.size; H = np.zeros((n, n))
    for k in range(n):
        d = np.zeros(n); d[k] = h
        Fp = force_fn((R.reshape(-1) + d).reshape(-1, 3)).reshape(-1)
        Fm = force_fn((R.reshape(-1) - d).reshape(-1, 3)).reshape(-1)
        H[:, k] = -(Fp - Fm) / (2 * h)
    return 0.5 * (H + H.T)


def freqs_thz(H):
    w2 = np.linalg.eigvalsh(H / MSI)                       # eV/(A^2 amu)
    conv = np.sqrt(units._e / units._amu) * 1e10 / (2 * np.pi) / 1e12   # sqrt(eV/A^2/amu) -> THz
    return np.sign(w2) * np.sqrt(np.abs(w2)) * conv


def main():
    tr, te = T.load_data()
    e0 = (tr['E'] / 8).mean().item()
    models = {}
    for head in ['energy', 'direct']:
        m = T.build('eq', 3.0, head=head, e0=e0, seed=0)
        cpu = T.train(m, tr, np.arange(400), steps=2500, bs=16, wE=10.0, wF=1.0, seed=0)
        ev = T.evaluate(m, te)
        models[head] = m
        g = M.Graph(te['R'].numpy(), te['cell'].numpy(), 3.0)
        Fp = M.predict(m, te['R'].reshape(-1, 3), te['cell'], g)['F']
        net = Fp.sum(1).norm(dim=-1)
        print(f"    {head:6s}: |sum_i F_i| over test cells: mean {net.mean().item()*1e3:.2e} meV/A, max {net.max().item()*1e3:.2e} meV/A")
        torch.save(m.state_dict(), os.path.join(T.DATA, f'102_nve_{head}.pt'))
        print(f"(A) {head:6s}: params {M.count_params(m)}  test E MAE {ev['E_mae']:.1f} meV/atom  "
              f"F MAE {ev['F_mae']:.0f} meV/A  train cpu {cpu:.0f} s", flush=True)
    oracle = MACECalculator(model_paths=os.path.join(T.DATA, 'mace_mp0_small.model'),
                            device='cpu', default_dtype='float64')
    at0 = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((2, 2, 2))
    res = {}
    for head in ['energy', 'direct']:
        for dt in [1.0, 0.5]:
            r = md(models[head], at0, dt, 5.0, oracle)
            t, E = np.array(r['t']), np.array(r['E'])
            ok = len(t) > 10
            sl = slope(t, E) if ok else np.nan
            slo = slope(np.array(r['to']), np.array(r['Eo'])) if len(r['to']) > 2 else np.nan
            fl = np.std(E - np.polyval(np.polyfit(t, E, 1), t)) if ok else np.nan
            Tm = np.mean(r['T'][len(r['T']) // 2:])
            print(f"(B) {head:6s} dt={dt:.1f} fs: t_end {t[-1]:.2f} ps | drift of E_model+K {sl:+8.3f} meV/atom/ps, "
                  f"fluct {fl:.3f} meV/atom | drift of E_oracle+K {slo:+8.3f} meV/atom/ps | "
                  f"<T> 2nd half {Tm:5.0f} K | max disp {r['disp']:.2f} A, centre-of-mass shift {r['com']:.2f} A | cpu {r['cpu']:.0f} s", flush=True)
            r['T'] = list(map(float, r['T'])); r['drift'] = sl; r['drift_oracle'] = slo; r['fluct'] = fl
            res[f'{head}_dt{dt}'] = r
    # (C) second derivatives
    at8 = bulk('Si', 'diamond', a=5.43, cubic=True)
    R8, C8 = at8.positions.copy(), at8.cell[:].copy()
    mc = models['energy']
    Had = hessian_ad(mc, R8, C8)
    g8 = M.Graph(R8[None], C8[None], 3.0, skin=0.3)
    Hfd = hessian_fd(lambda R: M.predict(mc, torch.as_tensor(R), torch.as_tensor(C8)[None], g8)['F'][0].numpy(), R8)
    print(f"(C) student Hessian: max |AD - FD| = {np.abs(Had-Hfd).max():.1e} eV/A^2 "
          f"(max |H| = {np.abs(Had).max():.2f}); asymmetry of AD Hessian {np.abs(Had-Had.T).max():.1e}")

    def f_or(R):
        a = Atoms('Si8', positions=R, cell=C8, pbc=True); a.calc = oracle
        return a.get_forces()
    t0 = time.process_time(); Hor = hessian_fd(f_or, R8); cpu_or = time.process_time() - t0
    fs, fo = freqs_thz(0.5 * (Had + Had.T)), freqs_thz(Hor)
    print(f"    oracle Hessian by FD: 48 force calls, cpu {cpu_or:.0f} s")
    print(f"    8-atom-cell modes (THz), highest four: student {np.round(fs[-4:],2)}  oracle {np.round(fo[-4:],2)}")
    print(f"    three acoustic (should be 0): student {np.round(fs[:3],3)}  oracle {np.round(fo[:3],3)}")
    print(f"    mean |student - oracle| over the 21 non-acoustic modes: {np.abs(fs[3:]-fo[3:]).mean():.2f} THz")
    res['freqs'] = {'student': fs.tolist(), 'oracle': fo.tolist()}
    json.dump(res, open(os.path.join(T.DATA, '102_nve.json'), 'w'))


if __name__ == '__main__':
    main()
