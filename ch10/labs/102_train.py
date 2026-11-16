"""Training and evaluation helpers shared by the Section 10.2 labs (one CPU thread, float64).

Loss (per batch), with n_at atoms per cell:
   L = wE * mean_b[ ((E_b - E_b^ref) / n_at)^2 ] + wF * mean_{b,i,a}[ (F - F^ref)^2 ] + wS * mean[(S - S^ref)^2]
Energies are compared per atom so that the weight does not change with cell size.
"""
import os, time, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
spec = importlib.util.spec_from_file_location('m102', os.path.join(HERE, '102_models.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
torch.set_num_threads(1)
DT = M.DT


def load_data():
    d = np.load(os.path.join(DATA, '102_si8.npz'))
    n = int(d['n_train'])
    T = {k: torch.as_tensor(d[k], dtype=DT) for k in ['R', 'cell', 'E', 'F', 'S']}
    tr = {k: v[:n] for k, v in T.items()}
    te = {k: v[n:] for k, v in T.items()}
    return tr, te


def build(kind, rc, head='energy', F=32, T=2, seed=0, e0=0.0):
    torch.manual_seed(seed)
    net = (M.InvNet if kind == 'inv' else M.EqNet)(rc=rc, F=F, T=T)
    return M.Potential(net, head=head, e0=e0)


def train(model, data, idx, steps=1500, bs=16, lr=3e-3, wE=1.0, wF=1.0, wS=0.0, seed=0, noiseF=0.0):
    """Adam with cosine decay. idx: indices of the training cells. noiseF: Gaussian label noise (eV/A)."""
    rng = np.random.default_rng(seed)
    g_all = M.Graph(data['R'][idx].numpy(), data['cell'][idx].numpy(), model.net.rc)
    R, C, E, F, S = (data[k][idx] for k in ['R', 'cell', 'E', 'F', 'S'])
    if noiseF > 0:
        F = F + torch.as_tensor(rng.normal(0, noiseF, F.shape), dtype=DT)
    n_at = R.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.process_time()
    n = len(idx)
    for it in range(steps):
        b = rng.choice(n, size=min(bs, n), replace=False)
        g = g_all.subset(b)
        out = M.predict(model, R[b].reshape(-1, 3), C[b], g, stress=wS > 0, create_graph=True)
        loss = wE * (((out['E'] - E[b]) / n_at) ** 2).mean()
        if wF > 0:
            loss = loss + wF * ((out['F'] - F[b]) ** 2).mean()
        if wS > 0:
            loss = loss + wS * ((out['S'] - S[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return time.process_time() - t0


def evaluate(model, data, chunk=50):
    n = data['R'].shape[0]; n_at = data['R'].shape[1]
    Ep, Fp = [], []
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        g = M.Graph(data['R'][sl].numpy(), data['cell'][sl].numpy(), model.net.rc)
        out = M.predict(model, data['R'][sl].reshape(-1, 3), data['cell'][sl], g)
        Ep.append(out['E']); Fp.append(out['F'])
    Ep = torch.cat(Ep); Fp = torch.cat(Fp)
    e_mae = ((Ep - data['E']) / n_at).abs().mean().item() * 1e3          # meV/atom
    f_mae = (Fp - data['F']).abs().mean().item() * 1e3                     # meV/A
    f_rmse = ((Fp - data['F']) ** 2).mean().sqrt().item() * 1e3
    return {'E_mae': e_mae, 'F_mae': f_mae, 'F_rmse': f_rmse}
