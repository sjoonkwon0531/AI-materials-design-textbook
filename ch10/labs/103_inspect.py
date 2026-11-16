"""Lab 10.3 (introspection): read architecture hyper-parameters directly from the loaded uMLIP objects.
Usage: python 103_inspect.py <model>  (re-executes under the right venv). Saves under key 'arch'."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('103_common'); model = sys.argv[1]; C.reexec_for(model)
calc, net, info = C.make_calc(model)
blocks = {}
if model == 'mace':
    sc = net.products[0].symmetric_contractions
    a = dict(cutoff=float(net.r_max), layers=int(net.num_interactions), sh_lmax=int(net.spherical_harmonics._lmax),
             hidden=str(net.interactions[0].hidden_irreps), correlation=int(sc.contractions[0].correlation),
             n_elements=len(net.atomic_numbers))
    a['body_order_per_layer'] = a['correlation'] + 1
    names = ['node_embedding', 'radial_embedding', 'interactions', 'products', 'readouts']
elif model == 'chgnet':
    g = net.graph_converter
    a = dict(cutoff=float(g.atom_graph_cutoff), bond_graph_cutoff=float(g.bond_graph_cutoff),
             layers=len(net.atom_conv_layers), bond_layers=len(net.bond_conv_layers),
             angle_layers=len(net.angle_layers), embed_dim=int(net.atom_embedding.embedding.embedding_dim),
             n_elements=int(net.atom_embedding.embedding.num_embeddings))
    names = ['atom_embedding', 'bond_basis_expansion', 'angle_basis_expansion', 'atom_conv_layers',
             'bond_conv_layers', 'angle_layers', 'site_wise', 'mlp']
elif model == 'sevennet':
    cf = calc.sevennet_config
    a = dict(cutoff=float(cf['cutoff']), layers=int(cf['num_convolution_layer']), lmax=int(cf['lmax']),
             hidden=cf['irreps_manual'][1], parity=bool(cf['is_parity']), n_elements=int(cf['_number_of_species']),
             radial=f"{cf['radial_basis']['radial_basis_name']} x{cf['radial_basis']['bessel_basis_num']}")
    names = None
elif model == 'mattersim':
    ma = net.model_args
    a = dict(backbone=type(net).__name__, cutoff=float(ma['cutoff']), threebody_cutoff=float(ma['threebody_cutoff']),
             layers=int(ma['num_blocks']), units=int(ma['units']), max_l=int(ma['max_l']), max_n=int(ma['max_n']),
             n_elements=int(ma['max_z']))
    names = ['rbf', 'sbf', 'edge_encoder', 'graph_conv', 'final', 'atom_embedding']
a['receptive_field_A'] = a['layers'] * a['cutoff']
import math
rho = 1 / 20.453                       # Si atoms per A^3 at the PBE (WIEN2k) volume of the Delta set
a['atoms_in_cutoff_Si'] = round(rho * 4 / 3 * math.pi * a['cutoff'] ** 3)
a['atoms_in_receptive_field_Si'] = round(rho * 4 / 3 * math.pi * a['receptive_field_A'] ** 3)
if names:
    tot = info['n_params']
    for n in names:
        blocks[n] = sum(p.numel() for p in getattr(net, n).parameters())
else:  # SevenNet: group by layer prefix
    for n, mod in net.named_children():
        key = n.split('_')[0] + '_' + n.split('_', 1)[1].split('_')[0] if n[0].isdigit() else n
        k = ('layer ' + n.split('_')[0]) if n[0].isdigit() else n
        blocks[k] = blocks.get(k, 0) + sum(p.numel() for p in mod.parameters())
    blocks = {k: v for k, v in blocks.items() if v > 0}
a['params_by_block'] = blocks
print(f"[{model}] {info['label']}: params {info['n_params']:,}, {info['dtype']}")
for k, v in a.items():
    if k != 'params_by_block': print(f"    {k:20s} {v}")
print('    params by block: ' + ', '.join(f'{k} {v:,}' for k, v in blocks.items()))
C.save(model, 'arch', a)
