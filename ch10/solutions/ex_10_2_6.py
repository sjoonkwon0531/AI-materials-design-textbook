"""Ex 10.2.6: direct-force model of Lab 10.3 with the net force projected out, F_i <- F_i - mean_j F_j.
Same NVE protocol as Lab 10.3 (64-atom Si, 600 K MB start seed 0, 5 ps, velocity Verlet).
Also: work around a closed loop in configuration space (curl test) for cons / direct / direct-projected.
Run with /tmp/claude-0/v10/bin/python."""
import os, sys, importlib.util, warnings
import numpy as np, torch
warnings.filterwarnings('ignore'); torch.set_num_threads(1)
LABS = '../ch10/labs'
spec = importlib.util.spec_from_file_location('n102', os.path.join(LABS, '102_nve.py'))
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)
T, M = N.T, N.M
from ase.build import bulk
from mace.calculators import MACECalculator
tr, te = T.load_data(); e0 = (tr['E'] / 8).mean().item()
models = {}
for head in ['energy', 'direct']:
    m = T.build('eq', 3.0, head=head, e0=e0, seed=0)
    m.load_state_dict(torch.load(os.path.join(T.DATA, f'102_nve_{head}.pt')))
    models[head] = m


class Projected(torch.nn.Module):
    """Wrapper: same model, forces with the mean removed."""
    def __init__(self, m):
        super().__init__(); self.m = m; self.net = m.net; self.head = m.head
orig_predict = M.predict


def predict_proj(model, R, cell, g, **kw):
    if isinstance(model, Projected):
        out = orig_predict(model.m, R, cell, g, **kw)
        out['F'] = out['F'] - out['F'].mean(1, keepdim=True)
        return out
    return orig_predict(model, R, cell, g, **kw)
M.predict = predict_proj
models['direct_proj'] = Projected(models['direct'])

# sanity: net force on test cells
g = M.Graph(te['R'].numpy(), te['cell'].numpy(), 3.0)
for k in ['direct', 'direct_proj']:
    F = M.predict(models[k], te['R'].reshape(-1, 3), te['cell'], g)['F']
    print(f"{k:12s}: mean |sum_i F_i| over test cells {F.sum(1).norm(dim=-1).mean().item()*1e3:.2e} meV/A, "
          f"test F MAE {(F-te['F']).abs().mean().item()*1e3:.1f} meV/A", flush=True)

# curl test: circulation of F around a small square loop in a random 2D plane of configuration space
at0 = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((2, 2, 2))
rng = np.random.default_rng(1)
R0 = at0.positions + rng.normal(0, 0.05, at0.positions.shape)
C = torch.as_tensor(at0.cell[:])[None]; g64 = M.Graph(R0[None], at0.cell[:][None], 3.0, skin=0.8)
u = rng.normal(size=R0.shape); u /= np.linalg.norm(u); w = rng.normal(size=R0.shape); w -= (w * u).sum() * u; w /= np.linalg.norm(w)
def circulation(model, a=0.05, n=200):
    tot = 0.0
    ts = (np.arange(n) + 0.5) / n
    path = [(lambda t: a * t * u, a * u), (lambda t: a * u + a * t * w, a * w),
            (lambda t: a * (1 - t) * u + a * w, -a * u), (lambda t: a * (1 - t) * w, -a * w)]
    for pos, d in path:
        for t in ts:
            F = M.predict(model, torch.as_tensor(R0 + pos(t)), C, g64)['F'][0].numpy()
            tot += (F * d).sum() / n
    return tot
for k in ['energy', 'direct', 'direct_proj']:
    print(f"circulation of F around a 0.05 A x 0.05 A loop: {k:12s} {circulation(models[k])*1e3:+.4f} meV", flush=True)

oracle = MACECalculator(model_paths=os.path.join(T.DATA, 'mace_mp0_small.model'), device='cpu', default_dtype='float64')
for dt in [1.0, 0.5]:
    r = N.md(models['direct_proj'], at0, dt, 5.0, oracle)
    t, E = np.array(r['t']), np.array(r['E'])
    sl = N.slope(t, E); slo = N.slope(np.array(r['to']), np.array(r['Eo']))
    fl = np.std(E - np.polyval(np.polyfit(t, E, 1), t)); Tm = np.mean(r['T'][len(r['T']) // 2:])
    print(f"direct_proj dt={dt:.1f} fs: t_end {t[-1]:.2f} ps | drift E_model+K {sl:+.3f} meV/atom/ps, fluct {fl:.3f} | "
          f"drift E_oracle+K {slo:+.3f} | <T> 2nd half {Tm:.0f} K | max disp {r['disp']:.2f} A, COM shift {r['com']:.3f} A | cpu {r['cpu']:.0f} s", flush=True)
