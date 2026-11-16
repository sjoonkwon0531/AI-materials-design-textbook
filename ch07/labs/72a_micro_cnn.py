"""Lab 7.2.1b -- 2D CNN on synthetic microstructures: translation is free, rotation is not.
Three classes of 64x64 binary 'micrographs' made by thresholding band-pass-filtered noise:
  0 lamellar (anisotropic filter, orientation theta), 1 bicontinuous labyrinth (isotropic, 50% area),
  2 droplets (isotropic, ~20% area).
Training lamellae only within +-15 deg of one orientation; test over all orientations.
Run: python 72a_micro_cnn.py   (about 3-4 min on one shared core)"""
import time, numpy as np, torch
from scipy.ndimage import rotate
torch.set_num_threads(1)
T0 = time.time(); L = 64
kx = np.fft.fftfreq(L)[:, None]; ky = np.fft.fftfreq(L)[None, :]
K = np.sqrt(kx ** 2 + ky ** 2); PHI = np.arctan2(ky, kx)

def micro(cls, rng, theta=None):
    k0 = rng.uniform(0.09, 0.14)                       # period ~ 7-11 px
    ring = np.exp(-0.5 * ((K - k0) / 0.02) ** 2)
    if cls == 0:                                      # lamellae: keep wavevectors near +-theta
        th = np.deg2rad(theta)
        ang = np.exp(-0.5 * (np.angle(np.exp(2j * (PHI - th))) / (2 * np.deg2rad(12))) ** 2)
        filt = ring * ang
    else:
        filt = ring
    f = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(L, L))) * filt))
    f = (f - f.mean()) / f.std()
    thr = 0.0 if cls < 2 else 0.85
    img = (f > thr).astype(np.float32)
    return img + 0.15 * rng.normal(size=(L, L)).astype(np.float32)   # detector noise

def dataset(n, rng, lam_range, aug=False):
    X, y, th = [], [], []
    for i in range(n):
        c = i % 3; t = rng.uniform(*lam_range) if c == 0 else 0.0
        img = micro(c, rng, t)
        if aug:                                       # rotation augmentation: arbitrary angle
            img = rotate(img, rng.uniform(0, 360), reshape=False, mode="wrap", order=1)
        X.append(img); y.append(c); th.append(t)
    return torch.tensor(np.stack(X))[:, None], torch.tensor(y), np.array(th)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, padding=2, padding_mode="circular"), torch.nn.SiLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 5, padding=2, padding_mode="circular"), torch.nn.SiLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 3, padding=1, padding_mode="circular"), torch.nn.SiLU())
        self.out = torch.nn.Linear(32, 3)
    def forward(self, x): return self.out(self.f(x).mean((-1, -2)))   # global average pool

def train(X, y, seed, epochs=12):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); net = CNN()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    for ep in range(epochs):
        perm = torch.randperm(len(y), generator=g)
        for b in range(0, len(y), 64):
            j = perm[b:b + 64]
            loss = torch.nn.functional.cross_entropy(net(X[j]), y[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return net

rng = np.random.default_rng(0)
Xa, ya, _ = dataset(1800, rng, (-15, 15))
Xb, yb, _ = dataset(1800, np.random.default_rng(0), (-15, 15), aug=True)   # same seeds, rotated copies
angles = np.arange(0, 91, 15)
test = {}
for a in angles:   # lamellae at fixed angle a (and its mirror -a), plus the two isotropic classes
    r = np.random.default_rng(100 + a)
    X, y, _ = dataset(300, r, (a, a)); test[a] = (X, y)

# --- equivariance of one random conv layer: translation exact, 90-degree rotation not ---
torch.manual_seed(0)
conv = torch.nn.Conv2d(1, 8, 5, padding=2, padding_mode="circular"); x = Xa[:8]
fx = conv(x)
e_t = ((conv(torch.roll(x, (5, 11), (-2, -1))) - torch.roll(fx, (5, 11), (-2, -1))).norm() / fx.norm()).item()
e_r = ((conv(torch.rot90(x, 1, (-2, -1))) - torch.rot90(fx, 1, (-2, -1))).norm() / fx.norm()).item()
print(f"random conv layer: relative error  translation (5,11) px: {e_t:.1e}   rotation 90 deg: {e_r:.2f}")

res = {}
for name, (X, y) in {"no augmentation": (Xa, ya), "rotation augmentation": (Xb, yb)}.items():
    R = []
    for seed in range(2):
        net = train(X, y, seed)
        row = []
        with torch.no_grad():
            for a in angles:
                Xt, yt = test[a]; p = net(Xt).argmax(1)
                lam = yt == 0
                row.append(((p[lam] == 0).float().mean().item(), (p[~lam] == yt[~lam]).float().mean().item()))
            # translation invariance of the trained classifier (global pooling + circular padding)
            Xt, yt = test[0]; l0 = net(Xt); l1 = net(torch.roll(Xt, (13, 29), (-2, -1)))
            tinv = (l0 - l1).abs().max().item()
        R.append(row)
    R = np.array(R).mean(0); res[name] = R
    print(f"--- {name} (training lamellae within +-15 deg; mean of 2 seeds; logit change under (13,29)-px roll: {tinv:.1e})")
    print("   lamella angle (deg): " + "".join(f"{a:7d}" for a in angles))
    print("   lamellae recall    : " + "".join(f"{v:7.3f}" for v in R[:, 0]))
    print("   other two classes  : " + "".join(f"{v:7.3f}" for v in R[:, 1]))
    print(f"   [time {time.time()-T0:.0f} s]")
ex = [micro(0, np.random.default_rng(7), 0.0), micro(0, np.random.default_rng(7), 60.0),
      micro(1, np.random.default_rng(8)), micro(2, np.random.default_rng(9))]
np.savez("72a_micro_cnn.npz", angles=angles, noaug=res["no augmentation"], aug=res["rotation augmentation"], ex=np.stack(ex))
print(f"wall {time.time()-T0:.0f} s")
