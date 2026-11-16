"""Lab 9.5 (b): FNO-2D surrogate for the 2D Navier-Stokes vorticity equation (nu = 1e-3, torus).
Learns the unit-time flow map  w(t) -> w(t + 1)  from 200 solver trajectories (4000 pairs),
then is rolled out autoregressively for 20 steps on 40 unseen initial conditions.
One thread, seed 0. Saves weights to ../data/95_fno_ns.pt and curves to ../data/95_fno_ns_results.npz."""
import time, importlib.util
import numpy as np
import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ns, fno = load("ns", "95_ns_solver.py"), load("fno", "95_fno2d.py")

torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
EPOCHS, BATCH, LR, NTRAIN = 4, 8, 2e-3, 200

d = np.load("../data/95_ns_nu1e-3_64.npz")
w = d["w"]                                               # (240, 21, 64, 64) float32
T = w.shape[1] - 1
train, test = w[:NTRAIN], w[NTRAIN:]
sigma = float(train.std())                               # one global scale; the map is not scale-free
x_tr = torch.from_numpy(train[:, :-1].reshape(-1, 64, 64) / sigma)
y_tr = torch.from_numpy(train[:, 1:].reshape(-1, 64, 64) / sigma)
print(f"train pairs {len(x_tr)}, test trajectories {len(test)}, sigma = {sigma:.3f}")

model = fno.FNO2d(modes=12, width=20, n_layers=4, proj=64)
print(f"FNO-2D: modes 12, width 20, 4 Fourier layers, {fno.n_params(model):,d} real parameters")
steps_total = EPOCHS * (len(x_tr) // BATCH)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps_total, pct_start=0.1)

def rollout(model, w0, nsteps):
    """Autoregressive rollout in physical units; w0: (B, 64, 64) numpy."""
    out = [torch.from_numpy(w0 / sigma)]
    with torch.no_grad():
        for _ in range(nsteps):
            out.append(model(out[-1]))
    return torch.stack(out, 1).numpy() * sigma

def one_step_error(model, traj):
    x = torch.from_numpy(traj[:, :-1].reshape(-1, 64, 64) / sigma)
    with torch.no_grad():
        p = torch.cat([model(x[i:i + 40]) for i in range(0, len(x), 40)]).numpy() * sigma
    return ns.rel_l2(p.reshape(traj.shape[0], T, 64, 64), traj[:, 1:])     # (B, T)

t0 = time.perf_counter(); c0 = time.process_time(); step = 0; hist = []
for ep in range(EPOCHS):
    perm = torch.randperm(len(x_tr)); run = 0.0
    for i in range(0, len(perm) - BATCH + 1, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = fno.rel_l2_loss(model(x_tr[idx]), y_tr[idx])
        loss.backward(); opt.step(); sched.step(); step += 1
        run += loss.item()
    e1 = one_step_error(model, test).mean()
    hist.append((ep + 1, time.perf_counter() - t0, run / (len(perm) // BATCH), e1))
    print(f"epoch {ep + 1}: train loss {hist[-1][2]:.4f} | test one-step rel. L2 {e1:.4f} | {hist[-1][1]:.0f} s", flush=True)
train_wall, train_cpu = time.perf_counter() - t0, time.process_time() - c0
torch.save({"state_dict": model.state_dict(), "sigma": sigma,
            "config": dict(modes=12, width=20, n_layers=4, proj=64)}, "../data/95_fno_ns.pt")

# ---------------------------------------------------------------- evaluation on 40 unseen trajectories
e_one = one_step_error(model, test)                       # teacher-forced: true w(t) -> w(t+1)
pred = rollout(model, test[:, 0], T)                      # free-running from w0
e_roll = ns.rel_l2(pred[:, 1:], test[:, 1:])
e_pers1 = ns.rel_l2(test[:, :-1], test[:, 1:])            # persistence, one step
e_pers0 = ns.rel_l2(np.repeat(test[:, :1], T, 1), test[:, 1:])   # "nothing happens" from t = 0
print(f"\ntraining: wall {train_wall:.0f} s on a shared machine, CPU time {train_cpu:.0f} s ({steps_total} Adam steps, batch {BATCH})")
print("  t      one-step     rollout      persistence(1 step)   persistence(from t=0)")
for t in (1, 2, 5, 10, 15, 20):
    print(f"  {t:2d}     {e_one[:, t - 1].mean():.4f}       {e_roll[:, t - 1].mean():.4f}       "
          f"{e_pers1[:, t - 1].mean():.4f}                {e_pers0[:, t - 1].mean():.4f}")
print(f"mean over t = 1..20: one-step {e_one.mean():.4f}, rollout {e_roll.mean():.4f};"
      f" rollout at t = 20: {e_roll[:, -1].mean():.4f} +/- {e_roll[:, -1].std() / np.sqrt(len(test)):.4f} (s.e., 40 samples),"
      f" worst sample {e_roll[:, -1].max():.4f}")

rms = lambda a: np.sqrt((a**2).mean((-2, -1))).mean()
print("rms of the rollout error / rms of w at t = 1, 5, 10, 20: "
      + ", ".join(f"{rms(pred[:, t] - test[:, t]):.4f} / {rms(test[:, t]):.3f}" for t in (1, 5, 10, 20)))
ae = np.abs(pred[:, -1] - test[:, -1]).mean(0)             # mean |error| per grid point at t = 20
edge = np.zeros((64, 64), bool); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
print(f"mean |error| at t = 20: on the outermost ring of grid points {ae[edge].mean():.4f}, elsewhere {ae[~edge].mean():.4f}"
      f"  (ratio {ae[edge].mean() / ae[~edge].mean():.2f})")

def enstrophy_spectrum(f):
    fh = np.abs(np.fft.fft2(f)) ** 2 / 64**4
    k = np.fft.fftfreq(64, d=1 / 64); KX, KY = np.meshgrid(k, k, indexing="ij")
    kb = np.rint(np.sqrt(KX**2 + KY**2)).astype(int).ravel()
    return np.array([np.bincount(kb, weights=s.ravel(), minlength=46) for s in fh.reshape(-1, 64, 64)]).mean(0)[:33]
spec_true, spec_pred = enstrophy_spectrum(test[:, -1]), enstrophy_spectrum(pred[:, -1])
spec_err = enstrophy_spectrum(pred[:, -1] - test[:, -1])
print("enstrophy spectrum at t = 20 (shells |k| = 1, 2, 4, 8, 16):")
print("  solver :", " ".join(f"{spec_true[k]:.2e}" for k in (1, 2, 4, 8, 16)))
print("  FNO    :", " ".join(f"{spec_pred[k]:.2e}" for k in (1, 2, 4, 8, 16)))
print(f"  share of the solver enstrophy above |k| = 12: {spec_true[13:].sum() / spec_true.sum():.2e};"
      f" FNO: {spec_pred[13:].sum() / spec_pred.sum():.2e}")
print(f"mean vorticity (conserved, = 0): max |<w>| over the FNO rollouts = {np.abs(pred.mean((2, 3))).max():.2e},"
      f" solver {np.abs(test.mean((2, 3))).max():.2e}")

np.savez("../data/95_fno_ns_results.npz", hist=np.array(hist), e_one=e_one, e_roll=e_roll, e_pers1=e_pers1,
         e_pers0=e_pers0, pred=pred[:4].astype(np.float32), truth=test[:4], spec_true=spec_true,
         spec_pred=spec_pred, spec_err=spec_err, train_wall=train_wall, train_cpu=train_cpu, sigma=sigma)
