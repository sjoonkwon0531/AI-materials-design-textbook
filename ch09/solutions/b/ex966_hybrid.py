"""Exercise 9.6.6: (a) hybrid rollout = pushforward FNO big step + n_c solver steps (dt = 0.05);
(b) effective conductivity (k1 = 1, k2 = 10) of the binarised microstructures with the periodic FEM of Ch.5 Lab 5.7.
Usage: python ex966_hybrid.py [smoke]"""
import sys, time, numpy as np, torch
import ch_common as cc
SMOKE = len(sys.argv) > 1
sol, test, rel = cc.sol, cc.test[:6] if SMOKE else cc.test, cc.rel
NT = len(test); NCS = (0, 1, 2, 5); DTB, dt = 2.5, 0.05
pf, cons, free = cc.lab_model("cons_pf"), cc.lab_model("cons"), cc.lab_model("free")
c25 = test[:, 1]
T0 = t0 = time.process_time()
# reference solver states at the physical times visited by each hybrid rollout: step counts j * (50 + nc)
need = sorted({j * (50 + nc) for nc in NCS for j in range(20)}); ref = {}; c, s = c25.copy(), 0
for n in need:
    c = sol.advance(c, n - s); s = n; ref[n] = c.copy()
print(f"reference solver: {need[-1]} steps from c(2.5) for {NT} trajectories, {time.process_time() - t0:.1f} CPU-s; check vs data at t = 50: {rel(ref[950], test[:, 20]).max():.1e}")

def hybrid(nc):
    out = [c25]; n_up = 0
    for _ in range(19):
        c = cc.apply(pf, out[-1]); F_raw = sol.free_energy(c)
        if nc: c = sol.advance(c, nc)
        n_up += int((F_raw > sol.free_energy(out[-1])).sum())           # increases produced by the surrogate step alone
        out.append(c)
    return np.stack(out, 1), n_up

m32 = cc.fno.FNO2d(modes=16, width=20, n_layers=4, proj=64, conserve=True)
m32.load_state_dict(torch.load(cc.DATA + "96_fno_ch_cons_pf.pt", weights_only=False)["state_dict"]); m32.eval()
x1 = torch.from_numpy(test[:1, 4].astype(np.float32)); c1 = test[:1, 4]
with torch.no_grad():
    ms_fno = 1e3 * cc.best(lambda: m32(x1))
ms_sol50 = 1e3 * cc.best(lambda: sol.advance(c1, 50))
ms_nc = {nc: (1e3 * cc.best(lambda: sol.advance(c1, nc)) if nc else 0.0) for nc in NCS}
print(f"cost (batch 1, one thread, smallest of 20): FNO big step {ms_fno:.2f} ms; solver 50 steps {ms_sol50:.2f} ms ({ms_sol50 / 50:.3f} ms per step);"
      f" solver n_c steps: " + ", ".join(f"{nc}: {v:.2f} ms" for nc, v in ms_nc.items() if nc))

print("\n(a) hybrid rollouts from the true c(2.5), 19 big steps; time of step j: 2.5 + j (2.5 + 0.05 n_c)")
print("  n_c  t_end   F increases   (surrogate step alone)   F(t_end) hybrid / solver, excess   F excess if labelled t=50   L(t_end) hyb / sol   slope (t>=5) hyb / sol   slope if labelled   err t~25   err t_end   max|c|   ms per big step   ms per unit time (solver "
      f"{ms_sol50 / DTB:.2f})")
H = {}
for nc in NCS:
    r, n_up = hybrid(nc); H[nc] = r
    tj = 2.5 + np.arange(20) * (DTB + dt * nc)
    R = np.stack([ref[j * (50 + nc)] for j in range(20)], 1)
    F, L, _ = cc.diagnostics(r); Fr, Lr, _ = cc.diagnostics(R); dF = np.diff(F, axis=1)
    F50, L50 = cc.F_s[:NT, -1].mean(), cc.L_s[:NT, -1].mean()
    e = rel(r[:, 1:], R[:, 1:])
    ms = ms_fno + ms_nc[nc]
    print(f"  {nc}    {tj[-1]:5.2f}   {int((dF > 0).sum()):3d} / {dF.size}     {n_up:3d} / {dF.size}                 {F[:, -1].mean():.3f} / {Fr[:, -1].mean():.3f}, {100 * (F[:, -1].mean() / Fr[:, -1].mean() - 1):+5.1f}%"
          f"            {100 * (F[:, -1].mean() / F50 - 1):+5.1f}%                {L[:, -1].mean():.3f} / {Lr[:, -1].mean():.3f}       {cc.slope(tj, L.mean(0)):.3f} / {cc.slope(tj, Lr.mean(0)):.3f}"
          f"           {cc.slope(cc.t1, L.mean(0)):.3f}              {e[:, 8].mean():.3f}      {e[:, 18].mean():.3f}       {np.abs(r).max():.3f}    {ms:.2f}              {ms / (DTB + dt * nc):.2f}", flush=True)
print(f"  solver at t = 50: F = {cc.F_s[:NT, -1].mean():.3f}, L = {cc.L_s[:NT, -1].mean():.3f}, slope {cc.slope(cc.t1, cc.L_s[:NT].mean(0)):.3f}")

print("\n(b) effective conductivity of the binarised microstructure (c > 0 -> k2 = 10, else k1 = 1), periodic Q1 FEM on 64 x 64 pixels")
homog = cc.load("homog", "../ch05/labs/55_homog.py")
cell = homog.PeriodicCell(64)
lam = np.zeros((64, 64), bool); lam[32:, :] = True; Kl = homog.k_eff(cell, lam, 1.0, 10.0)
print(f"  laminate check: k_xx = {Kl[0, 0]:.4f} (series 1.8182), k_yy = {Kl[1, 1]:.4f} (parallel 5.5)")
def keff(fields):
    t0 = time.process_time()
    k = np.array([np.trace(homog.k_eff(cell, c > 0, 1.0, 10.0)) / 2 for c in fields])
    return k, (fields > 0).mean((1, 2)), time.process_time() - t0
roll = {"cons": cc.rollout(cons, c25, 9)[:, -1], "free": cc.rollout(free, c25, 9)[:, -1], "cons_pf": H[0][:, 9]}
k_s, f_s, cpu = keff(test[:, 10])
print(f"  solver t = 25: k_eff = {k_s.mean():.3f} +- {k_s.std():.3f} (s.d. over {NT} trajectories), volume fraction of c > 0: {f_s.mean():.3f} +- {f_s.std():.3f}; {cpu / NT:.2f} CPU-s per microstructure")
A = np.stack([f_s - 0.5, np.ones(NT)], 1); co = np.linalg.lstsq(A, k_s, rcond=None)[0]
print(f"  solver microstructures: k_eff = {co[1]:.3f} + {co[0]:.2f} (f - 0.5), residual s.d. {np.std(k_s - A @ co):.3f};  d ln k / d f = {co[0] / co[1]:.2f}")
print("  microstructure            reference       rel. error of k_eff (mean |.|, mean signed, max)   |f - f_ref| (mean, signed)   pointwise rel. L2   corr(dk, df)   |dk| explained by df")
def row(name, fields, reff, k_ref, f_ref):
    k, f, _ = keff(fields); dk, df = (k - k_ref) / k_ref, f - f_ref
    pred = co[0] * df / k_ref                                               # what the volume-fraction error alone would give
    print(f"  {name:24s}  {reff:14s}  {100 * np.abs(dk).mean():5.2f}%  {100 * dk.mean():+5.2f}%  {100 * np.abs(dk).max():5.2f}%                 {np.abs(df).mean():.4f}  {df.mean():+.4f}"
          f"            {rel(fields, reff_fields).mean():.3f}             {np.corrcoef(dk, df)[0, 1]:+.2f}           {100 * np.abs(pred).mean():.2f}%  (residual {100 * np.abs(dk - pred).mean():.2f}%)", flush=True)
for name, fields in roll.items():
    reff_fields = test[:, 10]; row(name, fields, "solver t=25", k_s, f_s)
for nc in (1, 2, 5):
    reff_fields = ref[9 * (50 + nc)]; k_r, f_r, _ = keff(reff_fields)
    row(f"hybrid n_c = {nc}", H[nc][:, 9], f"solver t={2.5 + 9 * (DTB + dt * nc):.2f}", k_r, f_r)
# smoothing before thresholding: threshold sensitivity of the surrogate fields
for name, fields in (("solver", test[:, 10]), ("cons", roll["cons"]), ("cons_pf", roll["cons_pf"]), ("hybrid n_c = 2", H[2][:, 9])):
    print(f"  {name:16s}: share of pixels with |c| < 0.5: {(np.abs(fields) < 0.5).mean():.3f};  max |c| = {np.abs(fields).max():.3f}")
print(f"total {time.process_time() - T0:.0f} CPU-s")
