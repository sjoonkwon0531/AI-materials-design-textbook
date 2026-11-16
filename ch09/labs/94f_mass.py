"""§9.4.2 structure check: the pre-training family (rho = 0) conserves the mean of u exactly.
How well does the pre-trained FNO (no conservation built in) keep it?  Run: python 94f_mass.py  (seconds)"""
import sys, importlib, numpy as np, torch
sys.path.insert(0, "../ch09/labs")
C = importlib.import_module("94_common"); P = importlib.import_module("94b_pretrain")
m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH); m.load_state_dict(torch.load(C.D + "94_fno_member0.pt")); m.eval()
for name in ("pre_test", "lownu_test", "highk_test"):
    d = P.load(name); T_IDX = list(range(4, 41, 4))
    pred = C.predict(m, d, T_IDX).numpy(); true = d["u"][:, T_IDX]
    drift = np.abs(pred.mean(-1) - d["u"][:, :1].mean(-1))            # |mean_x u(t) - mean_x u0|, (N, 10)
    ref = np.abs(true.mean(-1) - d["u"][:, :1].mean(-1))
    rms = np.sqrt((true ** 2).mean())
    print(f"{name:11s}: solver max drift {ref.max():.1e} | FNO mean drift {drift.mean():.2e}, max {drift.max():.2e} "
          f"(rms u = {rms:.2f}; mean drift / rms error = {drift.mean() / np.sqrt(((pred - true) ** 2).mean()):.2f})")
