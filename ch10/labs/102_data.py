"""Shared data for the Section 10.2 labs: distorted diamond-Si cells labelled by an oracle.

The oracle (teacher) is MACE-MP-0 small. It stands in for DFT, because no DFT code that is fast
enough exists in this environment. Every lesson of Section 10.2 (force labels, stress, conservative
forces, angular resolution) concerns the student model and the loss, not the physics of the teacher,
so it transfers to real DFT labels.

Configurations: 8-atom cubic diamond-Si cells (a0 = 5.43 A), random symmetric strain
(|eps_ij| <= 0.03), Gaussian rattle with sigma drawn from U(0.02, 0.15) A.
Output: ../data/102_si8.npz with R (n,8,3), cell (n,3,3), E (n,), F (n,8,3), S (n,3,3) [eV/A^3],
and ../data/102_si8_{train,test}.xyz (extxyz) for the MACE command-line trainer.
Run with /tmp/claude-0/v10/bin/python.
"""
import os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
import torch
torch.set_num_threads(1)
from ase.build import bulk
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from mace.calculators import MACECalculator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
N_TRAIN, N_TEST = 400, 200


def make_configs(n, rng):
    out = []
    for _ in range(n):
        at = bulk('Si', 'diamond', a=5.43, cubic=True)
        eps = rng.uniform(-0.03, 0.03, (3, 3)); eps = 0.5 * (eps + eps.T)
        at.set_cell(at.cell[:] @ (np.eye(3) + eps), scale_atoms=True)
        sig = rng.uniform(0.02, 0.15)
        at.positions += rng.normal(0, sig, at.positions.shape)
        out.append(at)
    return out


def main():
    rng = np.random.default_rng(2026)
    calc = MACECalculator(model_paths=os.path.join(DATA, 'mace_mp0_small.model'),
                          device='cpu', default_dtype='float64')
    confs = make_configs(N_TRAIN + N_TEST, rng)
    t0 = time.process_time()
    R, C, E, F, S = [], [], [], [], []
    for at in confs:
        at.calc = calc
        e = at.get_potential_energy(); f = at.get_forces(); s = at.get_stress(voigt=False)
        at.calc = SinglePointCalculator(at, energy=e, forces=f, stress=s)
        R.append(at.positions.copy()); C.append(at.cell[:].copy()); E.append(e); F.append(f); S.append(s)
    cpu = time.process_time() - t0
    np.savez(os.path.join(DATA, '102_si8.npz'), R=np.array(R), cell=np.array(C), E=np.array(E),
             F=np.array(F), S=np.array(S), n_train=N_TRAIN)
    write(os.path.join(DATA, '102_si8_train.xyz'), confs[:N_TRAIN], format='extxyz')
    write(os.path.join(DATA, '102_si8_test.xyz'), confs[N_TRAIN:], format='extxyz')
    E = np.array(E); F = np.array(F)
    print(f"oracle calls: {len(confs)}   cpu {cpu:.1f} s   ({1e3*cpu/len(confs):.0f} ms per 8-atom cell)")
    print(f"E/atom: mean {E.mean()/8:.4f} eV, std of E/atom {E.std()/8*1e3:.1f} meV")
    print(f"|F| component mean {np.abs(F).mean():.3f} eV/A, max {np.abs(F).max():.2f} eV/A")
    # isolated-atom reference of the oracle (the E0 that MACE-MP-0 subtracts internally)
    from ase import Atoms
    a1 = Atoms('Si', positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=False); a1.calc = calc
    print(f"oracle isolated Si atom energy: {a1.get_potential_energy():.4f} eV")


if __name__ == '__main__':
    main()
