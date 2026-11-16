"""Lab 10.2 (102_stress): what each label is worth.
 (A) Stress as a strain derivative: autodiff stress vs central finite differences, for MACE-MP-0 small
     (its built-in stress) and for our from-scratch EqNet (Derivation Box in 10.2.1).
 (B) Energy-only vs energy+force training at equal numbers of labelled cells; force-only training and the
     undetermined energy constant; force-label noise; adding stress labels.
Student: EqNet, rc = 3.0 A, 2 layers, 32 channels; 1500 Adam steps, batch 8, lr 3e-3, seed 0.
Oracle data: ../data/102_si8.npz (MACE-MP-0 small labels standing in for DFT).
Run with /tmp/claude-0/v10/bin/python.
"""
import os, json, importlib.util, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t102', os.path.join(HERE, '102_train.py'))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
M = T.M
GPA = 160.21766                          # eV/A^3 -> GPa


def fd_stress(energy_of_cell, R, cell, h=1e-4):
    """sigma_ab = (1/V) dE/d eps_ab with symmetric eps, by central differences."""
    V = abs(np.linalg.det(cell)); S = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            e = np.zeros((3, 3)); e[a, b] += 0.5 * h; e[b, a] += 0.5 * h
            Ep = energy_of_cell(R @ (np.eye(3) + e), cell @ (np.eye(3) + e))
            Em = energy_of_cell(R @ (np.eye(3) - e), cell @ (np.eye(3) - e))
            S[a, b] = (Ep - Em) / (2 * h) / V
    return S


def part_a(tr, te):
    print("(A) stress: autodiff vs central finite differences (h = 1e-4), test cell 0")
    from ase import Atoms
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=os.path.join(T.DATA, 'mace_mp0_small.model'),
                          device='cpu', default_dtype='float64')
    R = te['R'][0].numpy(); C = te['cell'][0].numpy()

    def e_mace(Rx, Cx):
        at = Atoms('Si8', positions=Rx, cell=Cx, pbc=True); at.calc = calc
        return at.get_potential_energy()
    at = Atoms('Si8', positions=R, cell=C, pbc=True); at.calc = calc
    s_ad = at.get_stress(voigt=False); s_fd = fd_stress(e_mace, R, C)
    print(f"  MACE-MP-0 : sigma_xx {s_ad[0,0]*GPA:8.4f} GPa (AD) {s_fd[0,0]*GPA:8.4f} (FD); "
          f"max |AD-FD| over 9 components = {np.abs(s_ad-s_fd).max()*GPA:.1e} GPa")
    torch.manual_seed(0)
    m = T.build('eq', 3.0, e0=(tr['E'] / 8).mean().item())
    g = M.Graph(R[None], C[None], 3.0, skin=0.5)

    def e_student(Rx, Cx):
        return M.predict(m, torch.as_tensor(Rx), torch.as_tensor(Cx)[None], g)['E'].item()
    out = M.predict(m, torch.as_tensor(R), torch.as_tensor(C)[None], g, stress=True)
    s_ad = out['S'][0].numpy(); s_fd = fd_stress(e_student, R, C)
    print(f"  EqNet     : max |AD-FD| = {np.abs(s_ad-s_fd).max()*GPA:.1e} GPa  "
          f"(symmetric: max|S-S^T| = {np.abs(s_ad-s_ad.T).max()*GPA:.1e} GPa)")
    st = te['S'].numpy() * GPA
    print(f"  oracle stress over test cells: mean |sigma_ab| = {np.abs(st).mean():.2f} GPa")


def stress_mae(m, te):
    g = M.Graph(te['R'].numpy(), te['cell'].numpy(), m.net.rc)
    out = M.predict(m, te['R'].reshape(-1, 3), te['cell'], g, stress=True)
    return (out['S'] - te['S']).abs().mean().item() * GPA


def part_b(tr, te):
    print("(B) what the labels buy (test: 200 cells; E in meV/atom, F in meV/A)")
    e0 = (tr['E'] / 8).mean().item()
    res = {}

    def run(tag, N, **kw):
        m = T.build('eq', 3.0, e0=e0, seed=0)
        cpu = T.train(m, tr, np.arange(N), steps=1500, bs=8, seed=0, **kw)
        ev = T.evaluate(m, te)
        # energy error after removing the mean offset (the constant a force-only fit cannot fix)
        g = M.Graph(te['R'].numpy(), te['cell'].numpy(), 3.0)
        Ep = M.predict(m, te['R'].reshape(-1, 3), te['cell'], g)['E']
        d = (Ep - te['E']) / 8
        ev['E_mae_shifted'] = (d - d.mean()).abs().mean().item() * 1e3
        ev['E_offset'] = d.mean().item() * 1e3
        ev['S_mae'] = stress_mae(m, te); ev['cpu'] = cpu
        res[f'{tag}_N{N}'] = ev
        print(f"  {tag:22s} N={N:3d} | E MAE {ev['E_mae']:7.1f} (offset {ev['E_offset']:+7.1f}, after shift "
              f"{ev['E_mae_shifted']:5.1f}) | F MAE {ev['F_mae']:5.0f} | S MAE {ev['S_mae']:5.2f} GPa | cpu {cpu:4.0f} s",
              flush=True)
    for N in [10, 40, 160]:
        run(f'E only', N, wE=10.0, wF=0.0)
        run(f'E+F', N, wE=10.0, wF=1.0)
    run('F only', 160, wE=0.0, wF=1.0)
    run('E+F, F noise 0.2 eV/A', 160, wE=10.0, wF=1.0, noiseF=0.2)
    run('E+F+S', 40, wE=10.0, wF=1.0, wS=100.0)
    labels_per_cell = 1 + 3 * 8
    print(f"  labels per 8-atom cell: energy 1, forces {3*8}, stress 6 -> E+F carries {labels_per_cell}x the scalars of E only")
    json.dump(res, open(os.path.join(T.DATA, '102_stress.json'), 'w'), indent=1)


if __name__ == '__main__':
    tr, te = T.load_data()
    part_a(tr, te)
    part_b(tr, te)
