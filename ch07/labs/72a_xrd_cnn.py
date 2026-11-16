"""Lab 7.2.1a -- 1D CNN vs MLP on simulated powder XRD: structure-prototype identification
and robustness to a 2-theta zero shift (a translation of the pattern).
Ten prototypes (fcc, bcc, hcp, rocksalt, CsCl, zincblende, wurtzite, fluorite, rutile, cubic perovskite)
with random elements, lattice constants and axial ratios; pymatgen XRDCalculator, Cu K-alpha, 10-80 deg.
Run: python 72a_xrd_cnn.py   (generation ~1 min, cached; training a few min on one shared core)"""
import os, time, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
T0 = time.time()
PROTO = {  # name: (lattice kind, a-range (A), c/a, frac coords, species roles)
    "fcc":        ("cubic", (3.5, 4.2), None, [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], "AAAA"),
    "bcc":        ("cubic", (2.8, 3.4), None, [[0, 0, 0], [.5, .5, .5]], "AA"),
    "hcp":        ("hex", (2.5, 3.3), 1.63, [[1/3, 2/3, .25], [2/3, 1/3, .75]], "AA"),
    "rocksalt":   ("cubic", (4.0, 6.0), None, [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                                              [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]], "AAAABBBB"),
    "CsCl":       ("cubic", (3.0, 4.2), None, [[0, 0, 0], [.5, .5, .5]], "AB"),
    "zincblende": ("cubic", (5.3, 6.5), None, [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                                              [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]], "AAAABBBB"),
    "wurtzite":   ("hex", (3.0, 3.9), 1.63, [[1/3, 2/3, 0], [2/3, 1/3, .5], [1/3, 2/3, .375], [2/3, 1/3, .875]], "AABB"),
    "fluorite":   ("cubic", (5.0, 6.2), None, [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]] +
                   [[x, y, z] for x in (.25, .75) for y in (.25, .75) for z in (.25, .75)], "AAAA" + "B" * 8),
    "rutile":     ("tet", (4.3, 4.8), 0.64, [[0, 0, 0], [.5, .5, .5], [.305, .305, 0], [.695, .695, 0],
                                              [.805, .195, .5], [.195, .805, .5]], "AABBBB"),
    "perovskite": ("cubic", (3.7, 4.2), None, [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], "ABOOO"),
}
SYS = list(PROTO)
CAT = ["Li", "Na", "Mg", "Al", "K", "Ca", "Ti", "Fe", "Cu", "Zn", "Sr", "Ba", "Pb"]
AN = ["O", "S", "Se", "F", "Cl", "Br", "N"]
N_PER, TT = 400, np.arange(10.0, 80.0, 0.1)            # 700 channels
CACHE = "72a_xrd_peaks.npz"

def generate():
    from pymatgen.core import Structure, Lattice
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xc, rng = XRDCalculator(), np.random.default_rng(0)
    pk_x, pk_y, lab = [], [], []
    for ci, (name, (kind, ar, ca, fc, roles)) in enumerate(PROTO.items()):
        for _ in range(N_PER):
            a = rng.uniform(*ar)
            if kind == "cubic": lat = Lattice.cubic(a)
            elif kind == "hex": lat = Lattice.hexagonal(a, a * ca * rng.uniform(0.97, 1.03))
            else: lat = Lattice.tetragonal(a, a * ca * rng.uniform(0.97, 1.03))
            A, B = rng.choice(CAT), rng.choice(AN if "B" in roles else CAT)
            sp = [{"A": A, "B": B, "O": "O"}[r] for r in roles]
            p = xc.get_pattern(Structure(lat, sp, fc), two_theta_range=(TT[0], TT[-1]))
            pk_x.append(np.array(p.x)); pk_y.append(np.array(p.y)); lab.append(ci)
    print(f"generated {len(lab)} patterns in {time.time()-T0:.0f} s")
    np.savez(CACHE, x=np.array(pk_x, dtype=object), y=np.array(pk_y, dtype=object), lab=lab)

if not os.path.exists(CACHE): generate()
d = np.load(CACHE, allow_pickle=True); PX, PY, LAB = d["x"], d["y"], d["lab"]
print(f"patterns: {len(LAB)}, per class {np.bincount(LAB).tolist()}, median #peaks {int(np.median([len(x) for x in PX]))}")

def render(i, rng, shift=0.0):
    """Gaussian peaks, random FWHM 0.15-0.35 deg, 2% noise, max-normalized; zero shift adds to all 2theta."""
    w = rng.uniform(0.15, 0.35) / 2.3548
    prof = (PY[i][:, None] * np.exp(-0.5 * ((TT[None, :] - PX[i][:, None] - shift) / w) ** 2)).sum(0)
    prof = prof / prof.max() + 0.02 * rng.normal(size=TT.size)
    return prof.astype(np.float32)

rng = np.random.default_rng(1)
idx = rng.permutation(len(LAB)); n_te = 1000
te_i, tr_i = idx[:n_te], idx[n_te:]
Xtr = torch.tensor(np.stack([render(i, rng) for i in tr_i]))[:, None]; ytr = torch.tensor(LAB[tr_i])
yte = torch.tensor(LAB[te_i])
def test_set(shift, seed=2):
    r = np.random.default_rng(seed)
    return torch.tensor(np.stack([render(i, r, shift) for i in te_i]))[:, None]

class CNN(torch.nn.Module):
    def __init__(self, head="gap", deep=False):
        super().__init__()
        ch = [1, 16, 32, 32, 64, 64] if deep else [1, 16, 32, 32]
        layers = []
        for a_, b_ in zip(ch[:-1], ch[1:]):
            layers += [torch.nn.Conv1d(a_, b_, 7, padding=3), torch.nn.SiLU(), torch.nn.MaxPool1d(2)]
        self.f = torch.nn.Sequential(*layers); self.head = head
        self.out = torch.nn.Linear(ch[-1] if head == "gap" else ch[-1] * 87, len(SYS))
    def forward(self, x):
        h = self.f(x)                                     # (B, 32, 87): translation-equivariant feature map
        h = h.amax(-1) if self.head == "gap" else h.flatten(1)   # global max pool (invariant) or flatten (not)
        return self.out(h)
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(TT.size, 256), torch.nn.SiLU(),
                                     torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Linear(128, len(SYS)))
    def forward(self, x): return self.n(x)

def train(net, X, y, epochs=20, seed=0, aug_shift=0.0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(y) + 63) // 64))
    for ep in range(epochs):
        perm = torch.randperm(len(y), generator=g)
        for b in range(0, len(y), 64):
            j = perm[b:b + 64]; xb = X[j]
            if aug_shift > 0:   # random integer-channel roll as zero-shift augmentation
                k = int(torch.randint(-int(aug_shift / 0.1), int(aug_shift / 0.1) + 1, (1,), generator=g))
                xb = torch.roll(xb, k, -1)
            loss = torch.nn.functional.cross_entropy(net(xb), y[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net

def acc(net, X):
    with torch.no_grad(): return (net(X).argmax(1) == yte).float().mean().item()

# ---- exact translation equivariance of a conv layer (circular padding) vs aliasing from pooling ----
torch.manual_seed(0)
conv = torch.nn.Conv1d(1, 8, 9, padding=4, padding_mode="circular")
x0 = Xtr[:4]
e_conv = (conv(torch.roll(x0, 7, -1)) - torch.roll(conv(x0), 7, -1)).abs().max().item()
pool = torch.nn.MaxPool1d(2)
e_pool_even = (pool(torch.roll(x0, 8, -1)) - torch.roll(pool(x0), 4, -1)).abs().max().item()
e_pool_odd = (pool(torch.roll(x0, 7, -1)) - torch.roll(pool(x0), 3, -1)).abs().max().item()
print(f"equivariance check: |conv(shift x) - shift conv(x)|max = {e_conv:.1e} (circular padding, shift 7)")
print(f"   maxpool(2): shift 8 -> output shift 4: {e_pool_even:.1e};  shift 7 (odd) -> best integer shift 3: {e_pool_odd:.2f}")

shifts = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
Xte = {s: test_set(s) for s in shifts}
models = {"MLP": lambda: MLP(), "CNN+flatten": lambda: CNN("flat"), "CNN+global pool": lambda: CNN("gap"),
          "deep CNN+global pool": lambda: CNN("gap", deep=True)}
res = {}
print(f"train {len(tr_i)}, test {n_te}; 20 epochs AdamW + one-cycle; accuracy vs test zero shift (deg); 2 seeds")
print(f"{'model':28s} {'#params':>8s} " + "".join(f"{s:>7.1f}" for s in shifts))
for name, mk in list(models.items()) + [("MLP + shift augmentation", lambda: MLP())]:
    A = []
    for seed in range(2):
        net = train(mk(), Xtr, ytr, seed=seed, aug_shift=1.0 if "augment" in name else 0.0)
        A.append([acc(net, Xte[s]) for s in shifts])
    A = np.array(A); res[name] = A.mean(0)
    npar = sum(p.numel() for p in mk().parameters())
    print(f"{name:28s} {npar:8d} " + "".join(f"{a:7.3f}" for a in A.mean(0)) + f"   [time {time.time()-T0:.0f} s]")
    if name == "deep CNN+global pool":
        with torch.no_grad(): pred = net(Xte[0.0]).argmax(1).numpy()
        C = np.zeros((10, 10), int)
        for t_, p_ in zip(yte.numpy(), pred): C[t_, p_] += 1
        print("   confusion (rows true, cols pred; seed 1, no shift):", " ".join(s_[:4] for s_ in SYS))
        for i in range(10): print("   " + f"{SYS[i][:12]:12s}" + "".join(f"{v:5d}" for v in C[i]))
np.savez("72a_xrd_cnn.npz", shifts=shifts, names=np.array(list(res)), acc=np.array(list(res.values())),
         tt=TT, ex=np.stack([render(int(np.where(LAB == c)[0][0]), np.random.default_rng(5)) for c in range(10)]))
print(f"wall {time.time()-T0:.0f} s")
