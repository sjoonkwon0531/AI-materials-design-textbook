"""Lab 7.1.1 -- smooth vs ReLU activations for a 1D potential-energy curve.
Fit a Morse O-H-like curve with energy+force matching, then look at forces,
curvature (force constant) and energy conservation of NVE velocity-Verlet MD.
Run: python 71_activation.py  (about 1-2 min on one shared core)"""
import time, numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
t_start = time.time()
De, a, re = 4.0, 2.0, 1.0                 # eV, 1/A, A   (generic O-H-like Morse)
mu = 1.008 * 15.999 / (1.008 + 15.999)    # reduced mass, amu
CONV = 0.0096485                          # (eV/amu) -> A^2/fs^2
def morse(r):  e = np.exp(-a * (r - re)); return De * (1 - e) ** 2
def morse_f(r): e = np.exp(-a * (r - re)); return -2 * De * a * e * (1 - e)   # F = -dE/dr

r_tr = np.linspace(0.80, 1.60, 41)        # training grid, spacing 0.02 A
E_tr, F_tr = morse(r_tr), morse_f(r_tr)
acts = {"ReLU": torch.nn.ReLU, "GELU": torch.nn.GELU, "SiLU": torch.nn.SiLU, "tanh": torch.nn.Tanh}

def make(act, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1, 64), act(), torch.nn.Linear(64, 64), act(), torch.nn.Linear(64, 1))

def EF(net, r):
    r = torch.as_tensor(r, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    x = (r - 1.2) / 0.25                  # fixed input scaling
    E = net(x).sum(1)
    F = -torch.autograd.grad(E.sum(), r, create_graph=True)[0][:, 0]
    return E, F, r

def train(net, steps=4000):
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Et, Ft = torch.tensor(E_tr), torch.tensor(F_tr)
    for _ in range(steps):
        E, F, _ = EF(net, r_tr)
        loss = ((E - Et) ** 2).mean() + 0.1 * ((F - Ft) ** 2).mean()   # energy + force matching
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net

def verlet(net, r0, dt, T=2000.0):
    """1D NVE velocity Verlet for the bond length; returns max |E_tot - E_tot(0)| (eV)"""
    def ef(r):
        E, F, _ = EF(net, [r]); return E.item(), F.item()
    r, v = r0, 0.0; E, F = ef(r); E0 = E; dmax = 0.0
    for _ in range(int(T / dt)):
        v += 0.5 * dt * F / mu * CONV; r += dt * v
        E, F = ef(r); v += 0.5 * dt * F / mu * CONV
        dmax = max(dmax, abs(E + 0.5 * mu * v * v / CONV - E0))
    return dmax

rg = np.linspace(0.80, 1.60, 8001)        # dense evaluation grid, 1e-4 A
k_true = 2 * De * a * a
print(f"Morse De={De} eV, a={a} 1/A, re={re} A, mu={mu:.4f} amu; train on {len(r_tr)} points in [0.80,1.60] A (E and F)")
print(f"true force constant k = d2E/dr2 at re = {k_true:.2f} eV/A^2")
print(f"{'act':5s} {'seed':>4s} {'RMSE_E(meV)':>11s} {'RMSE_F(eV/A)':>12s} {'max|dF| per 1e-4A':>18s} {'k at re':>8s}")
store, dts = {}, [0.4, 0.2, 0.1, 0.05]
drift = {}
for name, act in acts.items():
    for seed in range(3):
        net = train(make(act, seed))
        E, F, r = EF(net, rg)
        k = torch.autograd.grad(-F.sum(), r)[0][:, 0].detach().numpy()   # d2E/dr2 via double backprop
        E, F = E.detach().numpy(), F.detach().numpy()
        rmse_e = np.sqrt(np.mean((E - morse(rg)) ** 2)) * 1e3
        rmse_f = np.sqrt(np.mean((F - morse_f(rg)) ** 2))
        jump = np.abs(np.diff(F)).max()
        k_re = k[np.argmin(abs(rg - re))]
        print(f"{name:5s} {seed:4d} {rmse_e:11.2f} {rmse_f:12.4f} {jump:18.4f} {k_re:8.2f}")
        if seed == 0:
            store[name] = (E, F, k)
            drift[name] = [verlet(net, 1.15, dt) for dt in dts]
print("NVE velocity-Verlet, r0 = 1.15 A at rest, 2000 fs: max |E_tot - E_tot(0)| (meV) vs dt (fs)  [seed 0]")
print("dt     " + "".join(f"{d:>10.2f}" for d in dts))
for name in acts:
    d = np.array(drift[name]) * 1e3
    slope = np.polyfit(np.log(dts), np.log(d), 1)[0]
    print(f"{name:6s} " + "".join(f"{x:10.4f}" for x in d) + f"   log-log slope {slope:5.2f}")
np.savez("71_activation.npz", rg=rg, E_true=morse(rg), F_true=morse_f(rg), dts=dts,
         **{f"E_{n}": store[n][0] for n in acts}, **{f"F_{n}": store[n][1] for n in acts},
         **{f"k_{n}": store[n][2] for n in acts}, **{f"drift_{n}": drift[n] for n in acts},
         r_tr=r_tr, F_tr=F_tr)
print(f"wall {time.time()-t_start:.0f} s")
