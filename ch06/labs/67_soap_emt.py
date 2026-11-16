"""Lab 6.7 step 4: when does structure matter, and how should a local descriptor be pooled?
Cu-Au fcc cells (32 atoms) whose composition AND chemical order vary; target = EMT energy
per atom (ase EMT, no relaxation). Compare composition-only features, a hand-built pair
feature, atom-averaged SOAP, and centre-species-resolved summed SOAP (a linear 'sum of
local energies' model)."""
import os, time, warnings, importlib.util
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.neighborlist import neighbor_list
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore"); t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("compat", os.path.join(HERE, "67_dscribe_compat.py"))
spec.loader.exec_module(importlib.util.module_from_spec(spec))          # ase>=3.26 shim for DScribe 2.1.2
from dscribe.descriptors import SOAP
calc = EMT()
def e_atom(at):
    at.calc = calc; return at.get_potential_energy() / len(at)
def make(rng, nAu, kind, strain):
    """32-atom fcc Cu-Au cell (2x2x2 conventional). Au placement rule:
    'random' - random sites; 'ordered' - alternate (001) planes filled first (L1_0-like);
    'segregated' - lowest-z sites first (Au-rich slab). Then 0-4 random antisite swaps,
    optional +-2% isotropic strain, and a 0.03 A rattle."""
    a = 3.61 + (4.08 - 3.61) * nAu / 32                      # Vegard lattice parameter (A)
    at = bulk("Cu", "fcc", a=a, cubic=True).repeat((2, 2, 2))
    z = at.get_scaled_positions()[:, 2]
    if kind == "random": order = rng.permutation(32)
    elif kind == "ordered": order = np.lexsort((rng.random(32), np.round(z * 4).astype(int) % 2))
    else: order = np.lexsort((rng.random(32), z))
    sym = np.array(["Cu"] * 32); sym[order[:nAu]] = "Au"
    for _ in range(int(rng.integers(0, 5))):
        if 0 < nAu < 32:
            i = rng.choice(np.where(sym == "Cu")[0]); j = rng.choice(np.where(sym == "Au")[0])
            sym[i], sym[j] = "Au", "Cu"
    at.set_chemical_symbols(sym)
    if strain: at.set_cell(at.cell * (1 + rng.uniform(-0.02, 0.02)), scale_atoms=True)
    at.rattle(0.03, seed=int(rng.integers(1e9)))
    return at
soap = SOAP(species=["Cu", "Au"], r_cut=5.0, n_max=6, l_max=4, sigma=0.5, periodic=True)
cv = KFold(5, shuffle=True, random_state=0)
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 3, 19)))
def krr_rbf(d):        # gamma grid scaled by 1/d so that the kernel is not degenerate in high dimension
    return lambda: GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                                {"kernelridge__alpha": [1e-6, 1e-4, 1e-2, 1], "kernelridge__gamma": [0.01 / d, 0.1 / d, 1 / d]}, cv=3)
def oof(model, F, y): return cross_val_predict(model(), F, y, cv=cv)
out = {}
for strain in (False, True):
    rng = np.random.default_rng(0)
    structs, x, kinds = [], [], []
    for k in range(400):
        nAu = int(rng.integers(0, 33)); kind = ("random", "ordered", "segregated")[k % 3]
        structs.append(make(rng, nAu, kind, strain)); x.append(nAu / 32); kinds.append(kind)
    y = np.array([e_atom(s) for s in structs]) * 1000; x = np.array(x); kinds = np.array(kinds)
    print("=== strain %s: N=%d cells; E/atom std %.1f meV" % ("+-2%" if strain else "off", len(y), y.std()))
    if not strain:
        for kd in ("ordered", "random", "segregated"):
            m = (kinds == kd) & (x > 0.3) & (x < 0.7)
            print("   %-10s cells with 0.3<x<0.7: n=%d, mean E/atom %.1f meV" % (kd, m.sum(), y[m].mean()))
        k5 = np.isclose(x, 0.5)
        print("   x=0.5 cells: E/atom from %.1f to %.1f meV (n=%d)" % (y[k5].min(), y[k5].max(), k5.sum()))
    t1 = time.time(); P = soap.create(structs, n_jobs=1)            # (N, 32, 390) per-atom SOAP
    print("   SOAP per atom: %d dims, %.1f ms per 32-atom cell" % (P.shape[2], 1e3 * (time.time() - t1) / len(P)))
    Z = np.array([s.numbers for s in structs])
    S_avg = P.mean(1)                                                # power spectra averaged over all centres ('outer')
    S_inn = SOAP(species=["Cu", "Au"], r_cut=5.0, n_max=6, l_max=4, sigma=0.5, periodic=True,
                 average="inner").create(structs, n_jobs=1)          # coefficients averaged first ('inner')
    S_res = np.concatenate([(P * (Z == 29)[..., None]).sum(1), (P * (Z == 79)[..., None]).sum(1)], 1) / 32
    pair = []
    for s in structs:
        i, j = neighbor_list("ij", s, 3.2); zz = s.numbers
        pair.append([np.mean(zz[i] != zz[j]), np.mean((zz[i] == 79) & (zz[j] == 79))])
    vol = np.array([s.get_volume() / 32 for s in structs])
    X1 = np.c_[x, x ** 2, x ** 3]; X2 = np.c_[X1, np.array(pair)]
    if strain: X1, X2 = np.c_[X1, vol], np.c_[X2, vol]
    print("   baseline (predict mean)                      MAE %6.2f meV/atom" % mean_absolute_error(y, np.full_like(y, y.mean())))
    res = {"comp": oof(ridge, X1, y), "comp_krr": oof(krr_rbf(X1.shape[1]), X1, y), "pair": oof(ridge, X2, y),
           "inner": oof(ridge, S_inn, y), "inner_krr": oof(krr_rbf(S_inn.shape[1]), S_inn, y),
           "avg": oof(ridge, S_avg, y), "res": oof(ridge, S_res, y)}
    lab = {"comp": "composition%s, cubic ridge" % (" + volume" if strain else ""), "comp_krr": "composition%s, RBF-KRR" % (" + volume" if strain else ""),
           "pair": "+ NN pair fractions (hand-built), ridge", "inner": "SOAP, DScribe average='inner', ridge", "inner_krr": "SOAP, DScribe average='inner', RBF-KRR",
           "avg": "SOAP, power spectra averaged ('outer'), ridge", "res": "SOAP summed per centre species, ridge"}
    for k, v in res.items(): print("   %-44s MAE %6.2f meV/atom" % (lab[k], mean_absolute_error(y, v)))
    out[strain] = (y, res, x, kinds)
# learning curve (no strain): composition vs centre-resolved SOAP
y, res, x, kinds = out[False]
rng = np.random.default_rng(1)
structs = None
print("learning curve (strain off; 80 held-out cells; 5 random draws):")
lc = []
# rebuild features for the strain-off set
rng0 = np.random.default_rng(0); st = []
for k in range(400):
    nAu = int(rng0.integers(0, 33)); st.append(make(rng0, nAu, ("random", "ordered", "segregated")[k % 3], False))
P = soap.create(st, n_jobs=1); Z = np.array([s.numbers for s in st])
S_res = np.concatenate([(P * (Z == 29)[..., None]).sum(1), (P * (Z == 79)[..., None]).sum(1)], 1) / 32
X1 = np.c_[x, x ** 2, x ** 3]
for n in (10, 20, 40, 80, 160, 320):
    r1, r2 = [], []
    for d in range(5):
        idx = rng.permutation(400); te, tr = idx[:80], idx[80:80 + n]
        r1.append(mean_absolute_error(y[te], ridge().fit(X1[tr], y[tr]).predict(X1[te])))
        r2.append(mean_absolute_error(y[te], ridge().fit(S_res[tr], y[tr]).predict(S_res[te])))
    lc.append([n, np.mean(r1), np.std(r1), np.mean(r2), np.std(r2)])
    print("   n_train=%3d  composition %.2f+/-%.2f   SOAP(centre-resolved) %.2f+/-%.2f meV/atom" % tuple(lc[-1]))
np.savez(os.path.join(HERE, "67_soap_emt.npz"), y=y, x=x, kind=kinds, lc=np.array(lc),
         oof_comp=res["comp"], oof_avg=res["avg"], oof_res=res["res"], oof_pair=res["pair"], oof_inner=res["inner"])
print("wall %.0f s" % (time.time() - t0))
