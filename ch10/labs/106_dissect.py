"""Lab 10.6 (b): dissecting MACE-MP-0 (small) with the vocabulary of Ch.7.

Part A  architecture read off the loaded model (irreps, l_max, correlation order, layers,
        cutoff, receptive field) and parameter count by block.
Part B  symmetry tests on a rattled, strained 40-atom SrTiO3 cell: energy invariance, force
        equivariance, stress transformation (sigma' = R sigma R^T), translation and permutation,
        in float64 and float32.
Part C  receptive field: move one atom of a 192-atom Si crystal (long axis 32.8 A) and see which forces change.
Part D  body order: many-body expansion of a 6-atom Si cluster, split into the energy read out
        after layer 1 and after layer 2 (forward hooks on the two readout blocks).

Run:  $MLIP_PY_A (MACE/CHGNet environment) 106_dissect.py      (CPU ~1-2 min, 1 thread)
"""
import os, sys, json, time, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from importlib import import_module
C = import_module('103_common'); C.reexec_for('mace')
import torch; torch.set_num_threads(1)
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from mace.calculators import MACECalculator
from scipy.spatial.transform import Rotation

DATA = os.path.join(os.path.dirname(HERE), 'data'); MP = os.path.join(DATA, 'mace_mp0_small.model')
calc64 = MACECalculator(model_paths=MP, device='cpu', default_dtype='float64')
net = calc64.models[0]
out = {}

# ------------------------------------------------------------------ Part A
print('=== Part A: architecture of MACE-MP-0 small ===')
it0, pr0 = net.interactions[0], net.products[0]
sc0 = pr0.symmetric_contractions.contractions[0]
rmax, T = float(net.r_max), int(net.num_interactions)
print(f'  elements: {len(net.atomic_numbers)}   cutoff r_max = {rmax} A   interaction layers T = {T}'
      f'   receptive field T*r_max = {T*rmax} A')
print(f'  radial basis: {net.radial_embedding.bessel_fn.bessel_weights.numel()} Bessel functions,'
      f' polynomial cutoff p = {float(net.radial_embedding.cutoff_fn.p)}')
print(f'  edge spherical harmonics: {it0.edge_attrs_irreps}   (l_max = {it0.edge_attrs_irreps.lmax})')
print(f'  node features passed between layers: {pr0.linear.irreps_out}')
print(f'  messages inside a layer (A-basis): {it0.conv_tp.irreps_out.simplify()}')
print(f'  correlation order nu = {sc0.correlation}  ->  body order of one layer = nu + 1 = {sc0.correlation+1}')
print(f'  avg_num_neighbors (message normalisation) = {it0.avg_num_neighbors:.2f}')
print(f'  readouts: {[type(r).__name__ for r in net.readouts]}')
blocks = {}
for name, mod in net.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n: blocks[name] = n
tot = sum(p.numel() for p in net.parameters())
skip = sum(sum(p.numel() for p in it.skip_tp.parameters()) for it in net.interactions)
radial = sum(sum(p.numel() for p in it.conv_tp_weights.parameters()) for it in net.interactions)
lin = sum(sum(p.numel() for p in it.linear.parameters()) + sum(p.numel() for p in it.linear_up.parameters())
          for it in net.interactions)
symc = sum(sum(p.numel() for p in pr.symmetric_contractions.parameters()) for pr in net.products)
print(f'  parameters total {tot:,}')
for k, v in blocks.items(): print(f'    {k:18s} {v:>10,}  ({100*v/tot:4.1f}%)')
print(f'    of which: element-dependent skip (self-connection) TP {skip:,} ({100*skip/tot:.1f}%),'
      f' symmetric contractions {symc:,} ({100*symc/tot:.1f}%),'
      f' radial MLPs {radial:,} ({100*radial/tot:.1f}%), equivariant linears {lin:,} ({100*lin/tot:.1f}%)')
out['A'] = dict(rmax=rmax, T=T, lmax=int(it0.edge_attrs_irreps.lmax), nu=int(sc0.correlation), total=tot,
                blocks=blocks, skip=skip, symc=symc, radial=radial, lin=lin)

# ------------------------------------------------------------------ Part B
print('\n=== Part B: symmetry tests, rattled + strained SrTiO3 2x2x2 (40 atoms) ===')
cub = Atoms('SrTiO3', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
            cell=np.eye(3) * 3.95, pbc=True).repeat((2, 2, 2))
rng = np.random.default_rng(0)
cub.set_cell(cub.cell @ (np.eye(3) + 0.02 * rng.normal(size=(3, 3))), scale_atoms=True)
cub.rattle(0.05, seed=1)
R = Rotation.random(random_state=7).as_matrix()


def efs(at, calc):
    a = at.copy(); a.calc = calc
    return a.get_potential_energy(), a.get_forces(), a.get_stress(voigt=False)


resB = {}
for dt in ('float64', 'float32'):
    calc = calc64 if dt == 'float64' else MACECalculator(model_paths=MP, device='cpu', default_dtype='float32')
    E, F, S = efs(cub, calc)
    rot = cub.copy(); rot.set_cell(cub.cell @ R.T, scale_atoms=True)          # rotate cell and atoms together
    Er, Fr, Sr = efs(rot, calc)
    tr = cub.copy(); tr.translate([0.37, -1.21, 2.05]); tr.wrap()
    Et, Ft, _ = efs(tr, calc)
    perm = rng.permutation(len(cub)); pm = cub[perm]
    Ep, Fp, _ = efs(pm, calc)
    fs = np.abs(F).max(); ss = np.abs(S).max()
    r = dict(dE_rot=abs(Er - E), dF_rot=np.abs(Fr - F @ R.T).max() / fs, dS_rot=np.abs(Sr - R @ S @ R.T).max() / ss,
             dE_tr=abs(Et - E), dF_tr=np.abs(Ft - F).max() / fs, dE_perm=abs(Ep - E), dF_perm=np.abs(Fp - F[perm]).max() / fs,
             sumF=np.abs(F.sum(0)).max())
    resB[dt] = r
    print(f'  {dt}: |dE| rotation {r["dE_rot"]:.1e} eV, translation {r["dE_tr"]:.1e} eV, permutation {r["dE_perm"]:.1e} eV'
          f' (E = {E:.4f} eV)')
    print(f'           force  rel. err: rotation (F -> R F) {r["dF_rot"]:.1e}, translation {r["dF_tr"]:.1e},'
          f' permutation {r["dF_perm"]:.1e};  |sum_i F_i| = {r["sumF"]:.1e} eV/A')
    print(f'           stress rel. err: rotation (S -> R S R^T) {r["dS_rot"]:.1e}   (max|F| {fs:.3f} eV/A, max|S| {ss*160.2177:.3f} GPa)')
out['B'] = resB

# ------------------------------------------------------------------ Part C
print('\n=== Part C: receptive field, Si 2x2x6 conventional cells (192 atoms, 10.9 x 10.9 x 32.8 A) ===')
si = bulk('Si', 'diamond', a=5.4646, cubic=True).repeat((2, 2, 6))  # MACE lattice constant (Lab part a)
si.rattle(0.01, seed=3)  # tiny rattle so that forces are not zero by symmetry
t0 = time.process_time()


def ef_node(at):
    a = at.copy(); a.calc = calc64
    F = a.get_forces(); return F, calc64.results['node_energy'].copy()


F0, e0n = ef_node(si)
si2 = si.copy(); si2.positions[0, 0] += 0.01
F1, e1n = ef_node(si2)
cpuC = time.process_time() - t0
dF = np.linalg.norm(F1 - F0, axis=1); de = np.abs(e1n - e0n)
r = si.get_distances(0, range(len(si)), mic=True)
bins = [(0, 0.1), (0.1, 6.0), (6.0, 12.0), (12.0, 19.0)]
resC = []
for lo, hi in bins:
    m = (r >= lo) & (r < hi)
    resC.append(dict(lo=lo, hi=hi, n=int(m.sum()), max_dF=float(dF[m].max()), max_de=float(de[m].max())))
    print(f'  {lo:4.1f} <= r < {hi:4.1f} A : {m.sum():4d} atoms, max |d eps_i| = {de[m].max():.2e} eV,'
          f' max |dF_i| = {dF[m].max():.2e} eV/A')
rlast_e = r[de > 1e-12].max(); rlast = r[dF > 1e-10].max()
print(f'  farthest atom with |d eps_i| > 1e-12 eV: r = {rlast_e:.3f} A; with |dF_i| > 1e-10 eV/A: r = {rlast:.3f} A')
print(f'  CPU for two 192-atom calls {cpuC:.1f} s')
np.savez(os.path.join(DATA, '106_receptive.npz'), r=r, dF=dF, de=de)
out['C'] = dict(bins=resC, rlast=rlast, rlast_e=rlast_e, cpu=cpuC)

# ------------------------------------------------------------------ Part D
print('\n=== Part D: many-body expansion of a 6-atom Si cluster ===')
grab = {}
hooks = [net.readouts[i].register_forward_hook(lambda m, inp, o, i=i: grab.__setitem__(i, o.detach().clone()))
         for i in range(2)]
scale = float(net.scale_shift.scale)


def layer_energies(at):
    a = at.copy(); a.calc = calc64; a.get_potential_energy()
    return [scale * float(grab[i][:, 0].sum()) for i in range(2)]


base = np.array([[0, 0, 1.6], [0, 0, -1.6], [1.6, 0, 0], [-1.6, 0, 0], [0, 1.6, 0], [0, -1.6, 0]])  # octahedron
rng = np.random.default_rng(11); pos = base + rng.normal(0, 0.15, base.shape)
clus = Atoms('Si6', positions=pos, cell=np.eye(3) * 30, pbc=False)
dmat = clus.get_all_distances(); print(f'  pair distances {dmat[np.triu_indices(6,1)].min():.2f}-{dmat.max():.2f} A (all inside r_max)')
lj = LennardJones(sigma=2.1, epsilon=0.1, rc=10.0)
Es = {}
for k in range(1, 7):
    for S in itertools.combinations(range(6), k):
        sub = clus[list(S)]
        l1, l2 = layer_energies(sub)
        a = sub.copy(); a.calc = lj
        Es[S] = (l1, l2, a.get_potential_energy())
for h in hooks: h.remove()


def mbe(idx):
    Ek = np.zeros(7)
    for S in Es:
        eps = sum((-1) ** (len(S) - len(T)) * Es[T][idx] for k in range(1, len(S) + 1) for T in itertools.combinations(S, k))
        Ek[len(S)] += eps
    return Ek


res = {name: mbe(i) for i, name in enumerate(['layer1', 'layer2', 'LJ'])}
print('  n-body increments E_n (eV), summed over all n-atom subsets:')
print('     n :' + ''.join(f'{n:>12d}' for n in range(1, 7)))
for name, Ek in res.items():
    print(f'  {name:6s}:' + ''.join(f'{x:12.2e}' for x in Ek[1:]) + f'   total {Ek[1:].sum():.4f}')
out['D'] = {k: v[1:].tolist() for k, v in res.items()}
json.dump(out, open(os.path.join(DATA, '106_dissect.json'), 'w'), indent=1, default=float)
print(f'\nCPU total {time.process_time():.1f} s')
