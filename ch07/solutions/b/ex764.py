import sys, torch; sys.path.insert(0, "../../../ch07/labs")
import importlib; eb = importlib.import_module("76_e3nn_block")
from e3nn import o3
for hid in ("16x0e + 8x1o + 8x1e", "16x0e + 8x1o"):
    m = eb.EquivariantPotential(irreps_hidden=hid)
    b2 = m.blocks[1]
    print(f"hidden {hid}: TP {b2.irreps_in} x {b2.irreps_sh} -> {b2.gate.irreps_in}")
    for ins in b2.tp.instructions:
        i1, i2, io = b2.tp.irreps_in1[ins.i_in1], b2.tp.irreps_in2[ins.i_in2], b2.tp.irreps_out[ins.i_out]
        print(f"   {i1} x {i2} -> {io}   weights {ins.path_shape}")
    print(f"   paths {len(b2.tp.instructions)}, TP weights/edge {b2.tp.weight_numel}, block params {sum(p.numel() for p in b2.parameters())}, model params {sum(p.numel() for p in m.parameters())}")
# pseudovector example: cross product of two bond vectors in ethanol; parity check
import numpy as np
d = np.load("../../../ch07/data/ethanol_xtb_300K.npz"); R = d["R"][2200]; Z = d["Z"]; print("Z", Z)
# --- are the 1e channels alive in the 2-layer model? (block-1 1e output, gradient of E wrt 1e-only weights)
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
species = {1: 0, 6: 1, 8: 2}
onehot1 = torch.nn.functional.one_hot(torch.as_tensor([species[z] for z in Z]), 3).double()
nm = 4; pos = torch.as_tensor(d["R"][:nm].reshape(-1, 3)); onehot = onehot1.repeat(nm, 1)
src, dst, batch = eb.full_graph(9, nm)
model = eb.EquivariantPotential()
feats = model.node_features(onehot, pos, src, dst)
irr = model.blocks[0].irreps_out
sl = irr.slices(); print("block-1 output irreps", irr, [str(x) for x in irr])
for (mul, ir), s in zip(irr, sl):
    print(f"   block-1 {mul}x{ir}: max |h| = {feats[0][:, s].abs().max().item():.2e}")
irr2 = model.blocks[1].irreps_out
for (mul, ir), s in zip(irr2, irr2.slices()):
    print(f"   block-2 {mul}x{ir}: max |h| = {feats[1][:, s].abs().max().item():.2e}")
E = model(onehot, pos, src, dst, batch, nm).sum(); E.backward()
# random perturbation of block-2 1e output does not change E (readout only l=0); check readout weight count
print("readout:", model.readout, "weights", model.readout.weight.numel())
