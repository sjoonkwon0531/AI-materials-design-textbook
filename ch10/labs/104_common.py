"""Common helpers for the Section 10.4 labs (owner C).

* oracle(): MACE-MP-0 small (float32) used as the stand-in for DFT.
* A periodic SchNet student (from Ch.7 labs/72b_schnet.py) with image shifts.
* batching, energy+force training, Langevin MD with a student, failure test.
Run under /tmp/claude-0/v10/bin/python (has mace, ase, dscribe).
"""
import os, time, math, warnings
import numpy as np
import torch
from importlib.machinery import SourceFileLoader

warnings.filterwarnings('ignore')
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sn = SourceFileLoader("sn", "../ch07/labs/72b_schnet.py").load_module()

A0 = 5.464          # MACE-MP-0 small equilibrium lattice constant of diamond Si (Lab 10.4.1 prints it)
RC = 4.0            # student cutoff (A): first and second neighbour shells of Si
FAIL_DMIN = 1.9     # an MD frame with a Si-Si distance below this is counted as a failure (A)


def oracle(dtype='float32'):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=os.path.join(DATA, 'mace_mp0_small.model'),
                          device='cpu', default_dtype=dtype)


def si64(a=A0):
    from ase.build import bulk
    return bulk('Si', 'diamond', a=a, cubic=True).repeat((2, 2, 2))


# ------------------------------------------------------------------ graphs and batches
def graph(pos, cell, rc=RC):
    from ase.neighborlist import primitive_neighbor_list
    i, j, S = primitive_neighbor_list('ijS', [True] * 3, cell, pos, rc)
    return np.stack([i, j]), S @ cell


def make_batch(frames, rc=RC):
    """frames: list of dicts with pos (n,3), cell (3,3), Z (n,), optional E, F."""
    Z, R, EI, SH, B, E, F = [], [], [], [], [], [], []
    off = 0
    for g, fr in enumerate(frames):
        ei, sh = graph(fr['pos'], fr['cell'], rc)
        EI.append(ei + off); SH.append(sh); Z.append(fr['Z']); R.append(fr['pos'])
        B.append(np.full(len(fr['Z']), g)); off += len(fr['Z'])
        if 'E' in fr:
            E.append(fr['E']); F.append(fr['F'])
    b = dict(Z=torch.tensor(np.concatenate(Z), dtype=torch.long),
             R=torch.tensor(np.concatenate(R), dtype=torch.float32),
             ei=torch.tensor(np.concatenate(EI, 1), dtype=torch.long),
             sh=torch.tensor(np.concatenate(SH), dtype=torch.float32),
             b=torch.tensor(np.concatenate(B), dtype=torch.long), n=len(frames),
             nat=torch.tensor([len(fr['Z']) for fr in frames], dtype=torch.float32))
    if E:
        b['E'] = torch.tensor(np.array(E), dtype=torch.float32)
        b['F'] = torch.tensor(np.concatenate(F), dtype=torch.float32)
    return b


def new_student(seed, e0):
    torch.manual_seed(seed)
    m = sn.SchNet(n_feat=32, n_rbf=16, cutoff=RC, n_inter=2)
    m.e_shift[14] = e0            # per-atom reference energy (E0) of Si
    return m


def predict(m, b, create_graph=False):
    return sn.energy_forces(m, b['Z'], b['R'], b['ei'], b['b'], b['n'], b['sh'], create_graph=create_graph)


def train(m, frames, epochs, lr=2e-3, wE=1.0, wF=10.0, bs=8, seed=0):
    """Energy (per atom) + force matching, Adam, cosine-decayed lr."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(m.parameters(), lr)
    nb = max(1, math.ceil(len(frames) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nb, eta_min=lr * 0.05)
    batches = None
    for ep in range(epochs):
        idx = rng.permutation(len(frames))
        for k in range(nb):
            sel = idx[k * bs:(k + 1) * bs]
            b = make_batch([frames[s] for s in sel])
            E, F = predict(m, b, create_graph=True)
            loss = wE * (((E - b['E']) / b['nat']) ** 2).mean() + wF * ((F - b['F']) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def errors(m, frames):
    """RMSE of energy per atom (meV/atom) and of force components (meV/A)."""
    out_e, out_f = [], []
    for k in range(0, len(frames), 16):
        b = make_batch(frames[k:k + 16])
        E, F = predict(m, b)
        out_e.append(((E - b['E']) / b['nat']).detach().numpy()); out_f.append((F - b['F']).detach().numpy().ravel())
    e = np.concatenate(out_e); f = np.concatenate(out_f)
    return float(1000 * np.sqrt((e ** 2).mean())), float(1000 * np.sqrt((f ** 2).mean()))


def forces_of(models, pos, cell, Z):
    b = make_batch([dict(pos=pos, cell=cell, Z=Z)])
    out = []
    for m in models:
        E, F = predict(m, b)
        out.append((E.item(), F.detach().numpy().astype(np.float64)))
    return out


def min_dist(pos, cell):
    from ase.neighborlist import primitive_neighbor_list
    d = primitive_neighbor_list('d', [True] * 3, cell, pos, 2.6)
    return d.min() if len(d) else 9.9


def langevin(model, pos0, cell, Z, T, nsteps, seed, dt_fs=2.0, gamma_fs=0.02, every=1, committee=None, keep_fail=False):
    """BAOAB Langevin MD driven by `model`. Returns frames, step of failure (or None).
    If `committee` is given, the max-atom force deviation is recorded every `every` steps."""
    kB = 8.617333262e-5                 # eV/K
    mass = 28.0855 * 103.6427            # amu -> eV fs^2 / A^2 (1 amu A^2/fs^2 = 103.6427 eV)
    rng = np.random.default_rng(seed)
    n = len(Z)
    v = rng.normal(0, np.sqrt(kB * T / mass), (n, 3)); v -= v.mean(0)
    x = pos0.copy()
    F = forces_of([model], x, cell, Z)[0][1]
    c1 = np.exp(-gamma_fs * dt_fs); c2 = np.sqrt((1 - c1 ** 2) * kB * T / mass)
    frames, fail = [], None
    for s in range(1, nsteps + 1):
        v += 0.5 * dt_fs * F / mass
        x += 0.5 * dt_fs * v
        v = c1 * v + c2 * rng.normal(size=(n, 3))
        x += 0.5 * dt_fs * v
        F = forces_of([model], x, cell, Z)[0][1]
        v += 0.5 * dt_fs * F / mass
        if not np.isfinite(x).all() or min_dist(x, cell) < FAIL_DMIN or np.abs(F).max() > 25.0:
            fail = s
            if keep_fail and np.isfinite(x).all():          # the configuration the student broke on
                fr = dict(pos=x.copy(), cell=cell, Z=Z, step=s, failed=True)
                if committee is not None:
                    Fs = np.array([o[1] for o in forces_of(committee, x, cell, Z)])
                    fr['dev'] = np.sqrt(((Fs - Fs.mean(0)) ** 2).sum(-1).mean(0)).max()
                frames.append(fr)
            break
        if s % every == 0:
            fr = dict(pos=x.copy(), cell=cell, Z=Z, step=s)
            if committee is not None:
                Fs = np.array([o[1] for o in forces_of(committee, x, cell, Z)])
                fr['dev'] = np.sqrt(((Fs - Fs.mean(0)) ** 2).sum(-1).mean(0)).max()   # DP-GEN max force deviation
            frames.append(fr)
    return frames, fail


def label(calc, frames):
    from ase import Atoms
    out = []
    for fr in frames:
        a = Atoms(numbers=fr['Z'], positions=fr['pos'], cell=fr['cell'], pbc=True); a.calc = calc
        out.append(dict(pos=fr['pos'].copy(), cell=fr['cell'], Z=fr['Z'],
                        E=float(a.get_potential_energy()), F=a.get_forces().copy()))
    return out


def save_frames(path, frames):
    np.savez_compressed(path, pos=np.array([f['pos'] for f in frames]), cell=np.array([f['cell'] for f in frames]),
                        Z=np.array([f['Z'] for f in frames]), E=np.array([f['E'] for f in frames]),
                        F=np.array([f['F'] for f in frames]),
                        tag=np.array([f.get('tag', '') for f in frames]),
                        T=np.array([f.get('T', 0) for f in frames]), step=np.array([f.get('step', 0) for f in frames]))


def load_frames(path):
    d = np.load(path, allow_pickle=False)
    return [dict(pos=d['pos'][k], cell=d['cell'][k], Z=d['Z'][k], E=float(d['E'][k]), F=d['F'][k],
                 tag=str(d['tag'][k]), T=float(d['T'][k]), step=int(d['step'][k])) for k in range(len(d['E']))]


from ase.calculators.calculator import Calculator, all_changes


class StudentCalc(Calculator):
    """ASE calculator wrapper for a trained student SchNet (energy and forces)."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, **kw):
        super().__init__(**kw); self.model = model

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        (E, F), = forces_of([self.model], atoms.positions, atoms.cell.array, atoms.numbers)
        self.results = dict(energy=E, forces=F)


def load_student(path, e0=0.0):
    m = new_student(0, e0); m.load_state_dict(torch.load(path)); m.eval(); return m
