"""Shared helpers for the Sec. 10.5 solution scripts: copied verbatim from labs/105_ft.py (functions, data, forgetting probes), without the lab driver."""
import os, sys, time, json, copy
import numpy as np, torch
sys.path.insert(0, '../../ch10/labs')
from importlib import import_module
C = import_module('104_common')
from ase import Atoms, units
from ase.build import bulk
from ase.eos import EquationOfState
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools

torch.set_num_threads(1)
t0 = time.process_time()
calc = C.oracle('float32')
base = calc.models[0]
keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
SI = calc.z_table.z_to_index(14)
train_all = C.load_frames(os.path.join(C.DATA, '105_xtb_train.npz'))
test = C.load_frames(os.path.join(C.DATA, '105_xtb_test.npz'))
ref = json.load(open(os.path.join(C.DATA, '105_xtb_meta.json')))['ref']


def to_data(fr):
    a = Atoms(numbers=fr['Z'], positions=fr['pos'], cell=fr['cell'], pbc=True)
    with torch_tools.default_dtype(torch.float32):
        cfg = mace_data.config_from_atoms(a, key_specification=keyspec, head_name=calc.head)
        return mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)


def run(model, frames, training=False):
    b = torch_geometric.batch.Batch.from_data_list([to_data(f) for f in frames])
    out = model(b.to_dict(), training=training, compute_force=True)
    return out['energy'], out['forces']


def errors(model, frames):
    E, F = run(model, frames)
    Er = torch.tensor([f['E'] for f in frames]); Fr = torch.tensor(np.concatenate([f['F'] for f in frames]), dtype=torch.float32)
    nat = torch.tensor([len(f['Z']) for f in frames], dtype=torch.float32)
    return (1000 * ((E.detach() - Er) / nat).pow(2).mean().sqrt()).item(), (1000 * (F.detach() - Fr).pow(2).mean().sqrt()).item()


def refit_e0(model, frames):
    """Shift the Si isolated-atom reference energy so that the mean per-atom energy matches."""
    E, _ = run(model, frames)
    d = np.mean([(f['E'] - e.item()) / len(f['Z']) for f, e in zip(frames, E)])
    model.atomic_energies_fn.atomic_energies[SI] += d
    return d


def eos(model):
    V, En = [], []
    for s in np.linspace(5.36, 5.66, 7):
        a = bulk('Si', 'diamond', a=s, cubic=True)
        fr = dict(pos=a.positions, cell=a.cell.array, Z=a.numbers)
        V.append(a.get_volume()); En.append(run(model, [fr])[0].item())
    v0, e0, B = EquationOfState(V, En).fit()
    return v0 ** (1 / 3), B / units.GPa


def finetune(model, frames, params, steps=40, lr=1e-3, bs=4, seed=0, wE=100.0, wF=1.0):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    rng = np.random.default_rng(seed)
    for s in range(steps):
        sel = rng.choice(len(frames), min(bs, len(frames)), replace=False)
        fr = [frames[k] for k in sel]
        E, F = run(model, fr, training=True)
        Er = torch.tensor([f['E'] for f in fr]); Fr = torch.tensor(np.concatenate([f['F'] for f in fr]), dtype=torch.float32)
        nat = torch.tensor([len(f['Z']) for f in fr], dtype=torch.float32)
        loss = wE * ((E - Er) / nat).pow(2).mean() + wF * (F - Fr).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# probe set for forgetting: other chemistries, rattled, labelled by the ORIGINAL MACE-MP-0
rng = np.random.default_rng(3)
probes = []
for name, a in [('C', bulk('C', 'diamond', a=3.57, cubic=True)), ('SiC', bulk('SiC', 'zincblende', a=4.38, cubic=True)),
                ('Ge', bulk('Ge', 'diamond', a=5.76, cubic=True)), ('Al', bulk('Al', 'fcc', a=4.04, cubic=True).repeat((2, 1, 1))),
                ('NaCl', bulk('NaCl', 'rocksalt', a=5.70, cubic=True))]:
    for k in range(3):
        b = a.copy(); b.positions += rng.normal(0, 0.06, b.positions.shape)
        probes.append((name, dict(pos=b.positions.copy(), cell=b.cell.array.copy(), Z=b.numbers.copy())))
F_orig = {}
for name, fr in probes:
    F_orig.setdefault(name, []).append(run(base, [fr])[1].detach().numpy())


def drift(model):
    out = {}
    for (name, fr) in probes:
        out.setdefault(name, []).append(run(model, [fr])[1].detach().numpy())
    return {n: 1000 * float(np.sqrt(np.mean([((a - b) ** 2).mean() for a, b in zip(out[n], F_orig[n])]))) for n in out}


