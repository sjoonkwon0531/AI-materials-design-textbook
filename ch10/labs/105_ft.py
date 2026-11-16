"""105_ft.py -- Lab for Section 10.5.3: fine-tuning MACE-MP-0 small to a different reference.

Target labels: GFN2-xTB on 8-atom Si cells (105_ft_data.py).  Custom PyTorch loop, float32, CPU.
Methods (same optimiser budget, 40 Adam steps, batch 4):
  zero-shot  : MACE-MP-0 small, only the Si reference energy E0 re-fitted (one number)
  readout    : E0 re-fit + train only the readout blocks (2,192 parameters)
  full       : E0 re-fit + train every parameter
  scratch    : same architecture, weights re-drawn from N(0,1) (the e3nn default), E0 re-fit, train all
plus a small SchNet trained from scratch to convergence (300 epochs) as a "cheap model" reference.
Measured: energy/force RMSE on 16 xTB test cells, EOS (a0, B) against xTB, and the drift of
forces on other materials (C, SiC, Ge, Al, NaCl) away from the original MACE-MP-0 (forgetting proxy).
"""
import os, sys, time, json, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


groups = {}
for n, p in base.named_parameters():
    groups[n.split('.')[0]] = groups.get(n.split('.')[0], 0) + p.numel()
print(f"MACE-MP-0 small: {sum(groups.values())} parameters {groups}; r_max {float(base.r_max):.1f} A, "
      f"{int(base.num_interactions)} interaction layers, hidden irreps {base.interactions[0].hidden_irreps}, "
      f"{len(calc.z_table.zs)} elements, E0(Si) {base.atomic_energies_fn.atomic_energies[SI].item():.4f} eV")
print(f"MPtrj (CHGNet paper numbers): {49295660 / 1580395:.1f} force vectors per configuration on average")
print(f"GFN2-xTB reference EOS: a0 {ref['a0']:.4f} A, B {ref['B_GPa']:.1f} GPa")
z0 = copy.deepcopy(base)
e_raw, f_raw = errors(z0, test)
a0_0, B_0 = eos(z0)
print(f"MACE-MP-0 small as is   : test E {e_raw:7.1f} meV/atom, F {f_raw:5.0f} meV/A | a0 {a0_0:.4f} A, B {B_0:5.1f} GPa")

results = []
for N in (4, 16, 48):
    tr = train_all[:N]
    for method in ('zero-shot', 'readout', 'full', 'scratch'):
        tm = time.process_time()
        m = copy.deepcopy(base)
        if method == 'scratch':
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for p in m.parameters():
                    p.copy_(torch.randn(p.shape, generator=g))
        d = refit_e0(m, tr)
        if method == 'readout':
            finetune(m, tr, [p for n, p in m.named_parameters() if n.startswith('readouts')], lr=5e-3)
        elif method == 'full':
            finetune(m, tr, list(m.parameters()), lr=1e-3)
        elif method == 'scratch':
            finetune(m, tr, list(m.parameters()), lr=1e-3)
        e, f = errors(m, test); a0, B = eos(m); dr = drift(m)
        results.append(dict(N=N, method=method, E=e, F=f, a0=float(a0), B=float(B), drift=dr, dE0=float(d), cpu=time.process_time() - tm))
        print(f"N={N:2d} {method:9s}: test E {e:6.2f} meV/atom, F {f:5.0f} meV/A | a0 {a0:.4f}, B {B:6.1f} GPa | "
              f"drift on C/SiC/Ge/Al/NaCl " + "/".join(f"{dr[k]:.0f}" for k in ('C', 'SiC', 'Ge', 'Al', 'NaCl')) +
              f" meV/A | dE0 {d:+.3f} eV | {time.process_time() - tm:.0f} s")

# longer optimiser budget at N = 48 (5x the steps)
for method in ('full', 'scratch'):
    tm = time.process_time()
    m = copy.deepcopy(base)
    if method == 'scratch':
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g))
    d = refit_e0(m, train_all)
    finetune(m, train_all, list(m.parameters()), steps=200, lr=1e-3)
    e, f = errors(m, test); a0, B = eos(m); dr = drift(m)
    results.append(dict(N=48, method=method + '-200', E=e, F=f, a0=float(a0), B=float(B), drift=dr, dE0=float(d), cpu=time.process_time() - tm))
    print(f"N=48 {method:7s} 200 steps: test E {e:6.2f} meV/atom, F {f:5.0f} meV/A | a0 {a0:.4f}, B {B:6.1f} GPa | "
          f"drift on C/SiC/Ge/Al/NaCl " + "/".join(f"{dr[k]:.0f}" for k in ('C', 'SiC', 'Ge', 'Al', 'NaCl')) +
          f" meV/A | {time.process_time() - tm:.0f} s")

# cheap-model reference: SchNet from scratch, converged on the same 48 labels
e0 = np.mean([f['E'] / 8 for f in train_all])
sn = C.new_student(0, e0); C.train(sn, train_all, epochs=300, seed=0)
es, fs = C.errors(sn, test)
Vs, Es_ = [], []
for sc in np.linspace(5.36, 5.66, 7):
    a = bulk('Si', 'diamond', a=sc, cubic=True); a.calc = C.StudentCalc(sn)
    Vs.append(a.get_volume()); Es_.append(a.get_potential_energy())
v0s, _, Bs = EquationOfState(Vs, Es_).fit()
print(f"SchNet from scratch, 48 labels, 300 epochs: test E {es:.2f} meV/atom, F {fs:.0f} meV/A | a0 {v0s ** (1 / 3):.4f}, B {Bs / units.GPa:.1f} GPa")
n_read = sum(p.numel() for n, p in base.named_parameters() if n.startswith('readouts'))
print(f"trainable parameters: readout {n_read}, full {sum(p.numel() for p in base.parameters())}")
json.dump(dict(results=results, raw=dict(E=e_raw, F=f_raw, a0=float(a0_0), B=float(B_0)), ref=ref, schnet=dict(E=float(es), F=float(fs), a0=float(v0s ** (1 / 3)), B=float(Bs / units.GPa))),
          open(os.path.join(C.DATA, '105_ft.json'), 'w'), indent=1)
print(f"CPU {time.process_time() - t0:.0f} s")
