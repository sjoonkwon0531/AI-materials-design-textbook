"""Exercise 10.5.3: force parity slope (through the origin) of MACE-MP-0 small against GFN2-xTB
on the 16 xTB test cells of Lab 10.9, before and after full fine-tuning (N = 48, 40 and 200 Adam steps).
Run: $MLIP_PY_A (MACE/CHGNet environment) ex_10_5_3.py"""
import time, copy
import numpy as np, torch
import ft_common as FT

torch.set_num_threads(1)
t0 = time.process_time()
test, train_all, base = FT.test, FT.train_all, FT.base
Fx = np.concatenate([f['F'] for f in test]).ravel()          # xTB force components


def slope(model):
    _, F = FT.run(model, test)
    Fm = F.detach().numpy().astype(np.float64).ravel()
    k = float(Fm @ Fx / (Fx @ Fx))                           # least squares through the origin: F_model = k F_xTB
    r = np.corrcoef(Fx, Fm)[0, 1]
    rms = 1000 * np.sqrt(np.mean((Fm - Fx) ** 2))
    resid = 1000 * np.sqrt(np.mean((Fm - k * Fx) ** 2))
    return k, r, rms, resid


print(f"xTB test force components: {Fx.size}, RMS |F| {1000*np.sqrt(np.mean(Fx**2)):.0f} meV/A, max |F| {abs(Fx).max():.2f} eV/A")
k, r, rms, res = slope(base)
print(f"MACE-MP-0 small as is      : slope {k:.3f}, Pearson r {r:.4f}, RMSE {rms:.0f} meV/A, RMSE after rescaling by 1/k {res/k:.0f} meV/A")
for steps in (40, 200):
    m = copy.deepcopy(base); FT.refit_e0(m, train_all)
    FT.finetune(m, train_all, list(m.parameters()), steps=steps, lr=1e-3)
    k, r, rms, res = slope(m)
    print(f"full fine-tune N=48 {steps:3d} st: slope {k:.3f}, Pearson r {r:.4f}, RMSE {rms:.0f} meV/A, residual about the fitted line {res:.0f} meV/A")
# what fraction of the zero-shot RMSE is removed by a single scale factor?
_, F = FT.run(base, test); Fm = F.detach().numpy().astype(np.float64).ravel()
k0 = float(Fm @ Fx / (Fx @ Fx))
print(f"zero-shot: RMSE of F_MACE/k0 vs xTB = {1000*np.sqrt(np.mean((Fm/k0 - Fx)**2)):.0f} meV/A (k0 = {k0:.3f})")
print(f"CPU {time.process_time() - t0:.0f} s")
