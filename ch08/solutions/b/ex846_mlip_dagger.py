"""Ex 8.4.6 (reduced): on-the-fly (DAgger-like) data for a 3-member SchNet-type MLIP ensemble.
Round 0 = the three Adam models of Lab 8.10 (N = 300, seeds 0-2).  Each round: ensemble-mean NVE MD (2 ps,
dt 0.5 fs) from the 300 K test frame 2200 with MB velocities seed 3; the first frame whose max-atom ensemble
force std exceeds 0.5 eV/A is labelled by GFN2-xTB (tblite) and added; every member is then FINE-TUNED
(300 steps, Adam lr 2e-3 -> cosine, batch 8) instead of retrained (compute budget).
Control: the same number of labels drawn at random from the 300 K xTB trajectory (frames not used for training)."""
import os, time, copy, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from tblite.ase import TBLite
from common import lab, LABS
torch.set_num_threads(1)
S = lab("83_mlip_sam")
t0 = time.time()
Z = S.Z
x0 = S.R3[2200].double().numpy()
THR = 0.5

def load_models():
    ms = []
    for s in range(3):
        m = S.Model(Z, S.Etr.mean().item(), S.Ftr.pow(2).mean().sqrt().item())
        m.load_state_dict(torch.load(os.path.join(LABS, f"83_mlip_adam_s{s}.pt"))); ms.append(m)
    return ms

class EnsCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, models):
        super().__init__(); self.models = [copy.deepcopy(m).double().eval() for m in models]; self.std = 0.0
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        R = torch.as_tensor(atoms.get_positions(), dtype=torch.float64)[None]
        Es, Fs = [], []
        for m in self.models:
            e, f = S.energy_forces(m, R); Es.append(e.item()); Fs.append(f[0].detach().numpy())
        Fs = np.array(Fs); self.std = float(np.sqrt(Fs.var(0).sum(-1)).max())
        self.results = {"energy": float(np.mean(Es)), "forces": Fs.mean(0)}

def md(models, nstep=4000, every=5):
    a = Atoms(numbers=Z, positions=x0.copy()); calc = EnsCalc(models); a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(3)); Stationary(a); ZeroRotation(a)
    d0 = a.get_all_distances(); bonded = (d0 > 0) & (d0 < 1.7)
    dyn = VelocityVerlet(a, 0.5 * units.fs); trig, trig_t, status, smax = None, None, "stable 2 ps", 0.0
    a.get_forces()
    for s in range(nstep // every):
        dyn.run(every); a.get_forces(); smax = max(smax, calc.std)
        if trig is None and calc.std > THR: trig, trig_t = a.get_positions().copy(), (s + 1) * every * 0.5e-3
        d = a.get_all_distances()
        if (d[bonded] > 2.0).any() or (d[np.triu_indices(9, 1)] < 0.6).any():
            status = "UNSTABLE %.2f ps" % ((s + 1) * every * 0.5e-3); break
    return status, trig, trig_t, smax, a.get_positions().copy()

xtb = TBLite(method="GFN2-xTB", verbosity=0)
def label(R):
    a = Atoms(numbers=Z, positions=R); a.calc = xtb
    return a.get_potential_energy(), a.get_forces()

def finetune(m, R, E, F, seed, steps=300, lr=2e-3, bs=8):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr * 0.01)
    for _ in range(steps):
        i = torch.randint(0, len(E), (bs,), generator=gen)
        opt.zero_grad(); S.loss_fn(m, R[i], E[i], F[i]).backward(); opt.step(); sch.step()
    return m

def f1000(models):
    return 1e3 * np.mean([S.evaluate(m, S.R10[S.i10], S.E10[S.i10], S.F10[S.i10])[1] for m in models])

def arm(kind, rounds=5, md_rounds=None):
    models = load_models(); R, E, F = S.Rtr.clone(), S.Etr.clone(), S.Ftr.clone()
    pool = list(S.perm[300:2000]); prng = np.random.default_rng(11)
    for k in range(rounds + 1):
        line = f"[{kind}] round {k}: N_train {len(E)} | F1000 MAE {f1000(models):5.0f} meV/A"
        if kind == "dagger" or k in (md_rounds or []):
            status, trig, tt, smax, Rend = md(models)
            line += f" | MD {status:<17s} max std {smax:5.2f} eV/A" + (f", trigger at {tt:.3f} ps" if tt else ", no trigger")
        print(line + f" | {time.time()-t0:.0f} s", flush=True)
        if k == rounds: break
        if kind == "dagger":
            Rn = trig if trig is not None else Rend
            e, f = label(Rn)
            Rn = torch.as_tensor(Rn, dtype=torch.float32)[None]; e = torch.tensor([e], dtype=torch.float32); f = torch.as_tensor(f, dtype=torch.float32)[None]
            print(f"      labelled frame: xTB E - E_min(train) = {float(e[0] - S.Etr.min()):.2f} eV, max |F| {float(f.norm(dim=-1).max()):.2f} eV/A", flush=True)
        else:
            j = int(pool.pop(int(prng.integers(len(pool)))))
            Rn, e, f = S.R3[j:j+1], S.E3[j:j+1], S.F3[j:j+1]
        R, E, F = torch.cat([R, Rn]), torch.cat([E, e]), torch.cat([F, f])
        models = [finetune(m, R, E, F, 100 * k + s) for s, m in enumerate(models)]

arm("dagger")
arm("random", md_rounds=[5])
print(f"wall {time.time()-t0:.0f} s")
