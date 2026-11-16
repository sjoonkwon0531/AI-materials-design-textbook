"""Ch.7.7 case study, step 2: short NVE MD (velocity Verlet, dt = 0.5 fs, 2 ps) with each trained model as an
ASE calculator, from the same 300 K test frame and velocities as a GFN2-xTB reference run.
Measures: stability (bond breaking / collapse), total-energy drift, angular-momentum drift, and the
O-H, C-C, C-O bond-length distributions against xTB. Writes 77_md.npz.  (~5-8 CPU-min.)"""
import os, sys, time, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np, torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from scipy.stats import wasserstein_distance
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("m77", os.path.join(HERE, "77_models.py"))
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)


class TorchCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model):
        super().__init__(); self.model = model.double().eval()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        R = torch.as_tensor(atoms.get_positions(), dtype=torch.float64)[None]
        E, F = M.energy_forces(self.model, R)
        self.results = {"energy": E.item(), "forces": F[0].detach().numpy()}


R3, E3, F3, Z = M.load(300)
x0 = R3[2200].double().numpy()
BONDS = {"O-H": (2, 3), "C-C": (0, 1), "C-O": (1, 2)}      # atom order: C(H3) C(H2) O H(O) H H H H H
NSTEP, EVERY = 4000, 5


def run(calc, label, seed=0):
    a = Atoms(numbers=Z, positions=x0.copy()); a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(seed))
    Stationary(a); ZeroRotation(a)
    d0 = a.get_all_distances(); bonded = (d0 > 0) & (d0 < 1.7)                 # covalent pairs at t = 0
    dyn = VelocityVerlet(a, 0.5 * units.fs)
    rec = {"Etot": [], "L": [], "T": [], "bonds": {k: [] for k in BONDS}}
    status, t0 = "stable", time.process_time()
    for s in range(NSTEP // EVERY):
        dyn.run(EVERY)
        d = a.get_all_distances()
        if (d[bonded > 0] > 2.0).any() or (d[np.triu_indices(9, 1)] < 0.6).any():
            status = "UNSTABLE at %.2f ps" % ((s + 1) * EVERY * 0.5e-3); break
        rec["Etot"].append(a.get_total_energy()); rec["T"].append(a.get_temperature())
        rec["L"].append(np.linalg.norm(a.get_angular_momentum()))
        for k, (i, j) in BONDS.items(): rec["bonds"][k].append(d[i, j])
    cpu = time.process_time() - t0
    Et = np.array(rec["Etot"]); t = np.arange(1, len(Et) + 1) * EVERY * 0.5e-3   # ps
    slope = np.polyfit(t, Et, 1)[0] if len(Et) > 10 else np.nan
    rec.update(status=status, drift=slope, Estd=Et.std() if len(Et) else np.nan, cpu=cpu, t=t)
    return rec


out = {}
from tblite.ase import TBLite
out["xTB"] = run(TBLite(method="GFN2-xTB", verbosity=0), "xTB")
out["xTB_seed1"] = run(TBLite(method="GFN2-xTB", verbosity=0), "xTB", seed=1)   # noise floor: other velocities
keys = ["mlp_N1000", "mlp_aug_N1000", "mlp_frame_N1000", "schnet_N1000", "nequip_N1000", "schnet_N100", "nequip_N100"]
keys += [k for k in ("schnet_N1000_long", "nequip_N1000_long") if os.path.exists(os.path.join(HERE, f"77_{k}.pt"))]
for key in keys:
    kind = key.split("_N")[0]
    w = M.Wrapped(kind, Z); w.load_state_dict(torch.load(os.path.join(HERE, f"77_{key}.pt")))
    out[key] = run(TorchCalc(w), key)
ref = out["xTB"]
print("NVE, dt 0.5 fs, %d steps (%.1f ps), start: 300 K test frame 2200 + MB(300 K, seed 0) velocities, P = L = 0" % (NSTEP, NSTEP * 0.5e-3))
print("%-18s %-22s %10s %9s %8s %8s | %s" % ("model", "status", "drift", "std(Etot)", "max|L|", "<T>", "bond mean (W1 vs xTB) [A]"))
for k, r in out.items():
    bs = "  ".join("%s %.3f (%.3f)" % (b, np.mean(v), wasserstein_distance(v, ref["bonds"][b])) for b, v in r["bonds"].items())
    print("%-18s %-22s %7.2f meV/ps %6.2f meV %8.4f %6.0f K | %s  [cpu %.0f s]" % (
        k, r["status"], 1e3 * r["drift"], 1e3 * r["Estd"], np.max(r["L"]) if len(r["L"]) else np.nan,
        np.mean(r["T"]) if len(r["T"]) else np.nan, bs, r["cpu"]))
np.savez(os.path.join(HERE, "77_md.npz"), **{f"{k}__{q}": (np.array(v) if q != "bonds" else np.array([v[b] for b in BONDS]))
                                          for k, r in out.items() for q, v in r.items() if q not in ("status",)},
         keys=np.array(list(out.keys())), status=np.array([r["status"] for r in out.values()]))
