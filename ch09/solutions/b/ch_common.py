"""Shared helpers for exercises 9.6.4 - 9.6.6 (same definitions as labs/96_fno_ch_eval.py)."""
import time, importlib.util
import numpy as np, torch
LABS = "../../../ch09/labs/"; DATA = "../../../ch09/data/"; OUT = "../../../ch09/solutions/b/"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ch, fno = load("ch", LABS + "96_ch_solver.py"), load("fno", LABS + "95_fno2d.py")
torch.set_num_threads(1)
_d = np.load(DATA + "96_ch_64.npz")
train, test, m_test = _d["train"], _d["test"].astype(float), _d["m_test"]
NREC = 20; t = 2.5 * np.arange(NREC + 1); t1 = t[1:]
sol = ch.CH2D(64)
rel = lambda a, b: np.sqrt(((a - b) ** 2).sum((-2, -1)) / (b ** 2).sum((-2, -1)))

def apply(model, x, bs=20):
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(x[i:i + bs])) for i in range(0, len(x), bs)]).numpy()

def rollout(model, c0, n):
    out = [c0]
    for _ in range(n):
        out.append(apply(model, out[-1]))
    return np.stack(out, 1)

def diagnostics(traj):
    F = np.stack([sol.free_energy(traj[:, i]) for i in range(traj.shape[1])], 1)
    L = np.stack([sol.length(traj[:, i]) for i in range(traj.shape[1])], 1)
    return F, L, np.abs(traj.mean((2, 3)) - traj[:, :1].mean((2, 3)))

def slope(tt, L, lo=5.0, hi=1e9):
    f = (tt >= lo) & (tt <= hi)
    return np.polyfit(np.log(tt[f]), np.log(L[f]), 1)[0]

def lab_model(tag):
    ck = torch.load(DATA + f"96_fno_ch_{tag}.pt", weights_only=False)
    m = fno.FNO2d(**ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return fno.to_double(m)

F_s, L_s, _ = diagnostics(test[:, 1:])

def evaluate(model, name, quench=True):
    """Same numbers as the Lab 9.20 output block for one model (model must already be float64)."""
    p1 = apply(model, test[:, :-1].reshape(-1, 64, 64)).reshape(test[:, 1:].shape); e1 = rel(p1, test[:, 1:])
    r = rollout(model, test[:, 1], NREC - 1); er = rel(r[:, 1:], test[:, 2:])
    F, L, drift = diagnostics(r); dF = np.diff(F, axis=1)
    F_in = np.stack([sol.free_energy(test[:, i]) for i in range(1, NREC)], 1)
    F_out = np.stack([sol.free_energy(p1[:, i]) for i in range(1, NREC)], 1)
    print(f"--- {name} ---")
    print("  one-step error (true input):   0->2.5: %.4f | 2.5->5: %.4f | 22.5->25: %.4f | 47.5->50: %.4f | mean of steps 2..20: %.4f"
          % (e1[:, 0].mean(), e1[:, 1].mean(), e1[:, 9].mean(), e1[:, 19].mean(), e1[:, 1:].mean()))
    print("  rollout from the true c(2.5):  t=5: %.4f | t=12.5: %.4f | t=25: %.4f | t=50: %.4f" % tuple(er[:, i].mean() for i in (0, 3, 8, 18)))
    print(f"  mass: max drift {drift.max():.2e};  free energy: increases along the rollouts {int((dF > 0).sum())} / {dF.size};"
          f" one step from true states {int((F_out > F_in).sum())} / {F_in.size}; F(50) = {F[:, -1].mean():.3f} (solver {F_s[:, -1].mean():.3f}, excess {100 * (F[:, -1].mean() / F_s[:, -1].mean() - 1):.0f}%)")
    print(f"  growth: L(25) = {L[:, 9].mean():.3f}, L(50) = {L[:, -1].mean():.3f} (solver {L_s[:, 9].mean():.3f}, {L_s[:, -1].mean():.3f});"
          f" slope on 5 <= t <= 50: {slope(t1, L.mean(0)):.3f} (solver {slope(t1, L_s.mean(0)):.3f});  max |c| = {np.abs(r).max():.3f}", flush=True)
    return r

def best(f, rep=20):
    b = np.inf
    for _ in range(rep):
        t0 = time.process_time(); f(); b = min(b, time.process_time() - t0)
    return b
