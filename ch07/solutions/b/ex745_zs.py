"""Ex 7.4.5(a): zero-shot force MAE of the G-pretrained model (GFN1 labels) on the 300 K and 1000 K (odd frames) test sets,
and GFN1-vs-GFN2 force MAE on the 1000 K pretraining frames (the level-of-theory gap)."""
import numpy as np, torch
src = open("../../../ch07/labs/74_pretrain_ethanol.py").read().replace("../../ch07/", "../../../ch07/")  # lab paths are relative to ch07/labs
exec(src.split("# ------------------------------------------------------------ pretraining (once per seed)")[0])
pre = torch.load("../../../ch07/labs/74_pretrained_states.pt")
t1k = np.load(DATA + "ethanol_xtb_1000K.npz"); odd = np.arange(1, 1000, 2)
R1k = torch.tensor(t1k["R"][odd], dtype=torch.float32); E1k = torch.tensor(t1k["E"][odd], dtype=torch.float32); F1k = torch.tensor(t1k["F"][odd], dtype=torch.float32)
m = SchNetDense(Z); m.load_state_dict(pre[("G", 0)]); set_scale(m, E1, F1)
print("G zero-shot F MAE: 300K %.1f, 1000K(odd) %.1f meV/A" % (evaluate(m, R300[test], E300[test], F300[test])[1], evaluate(m, R1k, E1k, F1k)[1]))
T = g1["T"]; F2_1k = torch.tensor(t1k["F"][::2], dtype=torch.float32)
print("GFN1 vs GFN2 force MAE on the 500 even 1000 K frames: %.1f meV/A" % ((F1[T == 1000] - F2_1k).abs().mean() * 1e3))
