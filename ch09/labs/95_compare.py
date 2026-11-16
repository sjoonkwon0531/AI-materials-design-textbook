"""Lab 9.5 (c): PINN vs FNO vs classical solver, accuracy and wall time, on equal terms.
Reference: 128^2 pseudo-spectral solution (dt = 2.5e-3) of the first 8 test initial conditions,
truncated to 64^2. Every method is scored against it on the 64^2 grid (coarse solutions are
zero-padded in Fourier space). Timings: one sample, one thread, same process, smallest CPU time of 10 calls."""
import re, time, importlib.util
import numpy as np
import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ns, fno = load("ns", "95_ns_solver.py"), load("fno", "95_fno2d.py")
torch.set_num_threads(1)

def best(f, rep=10):
    """Smallest CPU time of one call (process_time, so that other jobs on the machine are not counted)."""
    b = np.inf
    for _ in range(rep):
        t0 = time.process_time(); f(); b = min(b, time.process_time() - t0)
    return b

d = np.load("../data/95_ns_nu1e-3_64.npz")
test = d["w"][200:208].astype(float)                      # 8 test trajectories from the 64^2 solver
w0 = d["w0"][200:208]                                     # their initial conditions in float64
T = 20
t0 = time.process_time()
ref = ns.spectral_resample(ns.NS2D(128, dt=2.5e-3).trajectory(ns.spectral_resample(w0, 128), T, 400), 64)
print(f"reference 128^2, dt = 2.5e-3: CPU {time.process_time() - t0:.0f} s for 8 samples to T = {T}")
err = lambda a, t: ns.rel_l2(a, ref[:, t]).mean()

print("\n(1) classical solver at decreasing resolution (query: w0 -> w(t), error on the 64^2 grid)")
print("    N    dt       err t=1    err t=5    err t=20    ms per unit time (1 sample)")
rows = []
for N, dt in [(64, 1e-2), (48, 1e-2), (32, 2e-2), (24, 2e-2), (16, 4e-2), (12, 4e-2), (8, 4e-2)]:
    sol = ns.NS2D(N, dt=dt)
    tr = ns.spectral_resample(sol.trajectory(ns.spectral_resample(w0, N), T, int(round(1 / dt))), 64)
    w1 = ns.spectral_resample(w0[:1], N)
    ms = 1e3 * best(lambda: sol.advance(w1, int(round(1 / dt))))
    rows.append((N, dt, err(tr[:, 1], 1), err(tr[:, 5], 5), err(tr[:, 20], 20), ms))
    print(f"    {N:<3d}  {dt:.0e}    {rows[-1][2]:.2e}   {rows[-1][3]:.2e}   {rows[-1][4]:.2e}    {ms:6.2f}")
rows = np.array(rows)

print("\n(2) FNO-2D (Lab 9.5 (b)), autoregressive from w0")
ck = torch.load("../data/95_fno_ns.pt")
model = fno.FNO2d(**ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval(); sigma = ck["sigma"]
x = torch.from_numpy((w0 / sigma).astype(np.float32)); out = [x]
with torch.no_grad():
    for _ in range(T):
        out.append(model(out[-1]))
    pred = torch.stack(out, 1).numpy().astype(float) * sigma
    x1 = x[:1]
    ms_fno = 1e3 * best(lambda: [model(x1) for _ in range(5)], 20) / 5
e_fno = np.array([err(pred[:, t], t) for t in range(1, T + 1)])
print(f"    err t=1: {e_fno[0]:.2e}, t=5: {e_fno[4]:.2e}, t=20: {e_fno[19]:.2e};  {ms_fno:.2f} ms per unit time (1 sample)")
kf = np.fft.fftfreq(64, d=1 / 64); KXf, KYf = np.meshgrid(kf, kf, indexing="ij")
high = np.maximum(np.abs(KXf), np.abs(KYf)) >= 12          # modes the Fourier layers do not touch
share = lambda a: (np.abs(np.fft.fft2(a))[..., high] ** 2).sum() / (np.abs(np.fft.fft2(a)) ** 2).sum()
print(f"    modes outside the FNO band (max(|kx|,|ky|) >= 12): share of ||w0||^2 = {share(w0):.2e}, of ||w(1)||^2 = {share(ref[:, 1]):.2e};"
      f" share of the squared first-step error = {share(pred[:, 1] - ref[:, 1]):.2f}, of the error at t = 20 = {share(pred[:, 20] - ref[:, 20]):.2f}")
res = np.load("../data/95_fno_ns_results.npz")
gen_s = float(re.search(r"([\d.]+) s CPU", open("95_ns_data.out").read()).group(1))
train_s = float(res["train_cpu"])
print(f"    one-off cost (CPU time): data generation {gen_s:.0f} s (240 trajectories) + training {train_s:.0f} s = {gen_s + train_s:.0f} s")

print("\n(3) PINN for one instance (test trajectory 0), horizon T = 5, same CPU budget as the whole FNO training")
pinn = {}
for mode in ("soft", "hard"):
    p = np.load(f"../data/95_pinn_ns_T5_{mode}.npz")
    e = np.array([ns.rel_l2(p["pred"][t].astype(float), ref[0, t]) for t in range(1, 6)])
    pinn[mode] = (e, float(p["cpu"]), p["pred"])
    print(f"    {mode} IC: err t=1..5: " + " ".join(f"{v:.3f}" for v in e) + f";  CPU time {float(p['cpu']):.0f} s for this one instance")
print(f"    same instance: FNO err t=1: {ns.rel_l2(pred[0, 1], ref[0, 1]):.3f}, t=5: {ns.rel_l2(pred[0, 5], ref[0, 5]):.3f};"
      f" 64^2 solver t=5: {ns.rel_l2(test[0, 5], ref[0, 5]):.1e};  'nothing happens' (w = w0) t=1: {ns.rel_l2(w0[0], ref[0, 1]):.3f}, t=5: {ns.rel_l2(w0[0], ref[0, 5]):.3f}")

print("\n(4) break-even number of queries (one query = one initial condition advanced to T = 20)")
q_fno = T * ms_fno / 1e3
for N, dt, _, _, e20, ms in rows:
    q_sol = T * ms / 1e3
    tag = f"solver {int(N)}^2 (err {e20:.1e})"
    if q_sol > q_fno:
        print(f"    vs {tag:<28s}: {q_sol * 1e3:7.1f} ms vs FNO {q_fno * 1e3:.1f} ms per query -> speed-up {q_sol / q_fno:5.1f}x,"
              f" break-even N* = {(gen_s + train_s) / (q_sol - q_fno):,.0f} queries")
    else:
        print(f"    vs {tag:<28s}: {q_sol * 1e3:7.1f} ms vs FNO {q_fno * 1e3:.1f} ms per query -> the solver is {q_fno / q_sol:.1f}x faster; no break-even")
ok = rows[rows[:, 4] <= e_fno[19]]
m = ok[np.argmin(ok[:, 5])]
print(f"    matched accuracy at t = 20 (FNO error {e_fno[19]:.2e}): coarsest solver at least as accurate is {int(m[0])}^2"
      f" (error {m[4]:.1e}, {m[5]:.2f} ms per unit time vs FNO {ms_fno:.2f} ms)")

print("\n(5) heat equation of Lab 9.5 (a): PINN vs Crank-Nicolson")
h = np.load("../data/95_dde_heat_results.npz")
cn = h["cn"]
for k, name in (("soft", "soft-constraint PINN"), ("hard", "hard-constraint PINN")):
    st, wall, rel, emax = h[f"{k}_summary"][-1]
    j = np.argmax(cn[:, 2] <= rel)                        # coarsest CN grid with rel. L2 error below the PINN's
    print(f"    {name}: rel. L2 {rel:.2e} in {wall:.0f} s;  CN with n = {int(cn[j, 0])} reaches {cn[j, 2]:.2e} in {cn[j, 3]:.2f} ms"
          f"  -> ratio {wall / (cn[j, 3] / 1e3):,.0f}x")
np.savez("../data/95_compare.npz", rows=rows, e_fno=e_fno, ms_fno=ms_fno, gen_s=gen_s, train_s=train_s,
         e_pinn_soft=pinn["soft"][0], e_pinn_hard=pinn["hard"][0], pinn_cpu_soft=pinn["soft"][1], pinn_cpu_hard=pinn["hard"][1],
         ref0=ref[0], pred0=pred[0], pinn_pred_hard=pinn["hard"][2])
