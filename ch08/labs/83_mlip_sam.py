"""Lab 8.3 (c): does SAM help a small SchNet-type MLIP extrapolate from 300 K to 600 K / 1000 K ethanol?
Same data, split and SchNet-like model as Ch.7 Lab 7.21 (copied from ch7/labs/77_models.py), N = 300 frames,
1200 steps, batch 8, cosine schedule; Adam vs SAM(Adam) vs ASAM(Adam), 3 seeds each.
Measures force MAE at 300/600/1000 K and lambda_max of the training loss (energy+force, Lanczos on HVPs through the
double-backward graph, float64 copy).  Usage: python 83_mlip_sam.py   (~10-15 min on one thread under load)"""
import os, sys, copy, math, time, json, importlib.util
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("sam83", os.path.join(HERE, "83_sam.py"))
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)
DATA = os.path.join(HERE, "..", "..", "ch7", "data")
NAT, SPECIES = 9, {1: 0, 6: 1, 8: 2}
t0 = time.time()


def load(T):
    d = np.load(os.path.join(DATA, f"ethanol_xtb_{T}K.npz"))
    return (torch.as_tensor(d["R"], dtype=torch.float32), torch.as_tensor(d["E"], dtype=torch.float32),
            torch.as_tensor(d["F"], dtype=torch.float32), d["Z"])


def ssp(x): return nn.functional.softplus(x) - math.log(2.0)
class SSP(nn.Module):
    def forward(self, x): return ssp(x)


class SchNetLike(nn.Module):                       # identical to ch7/labs/77_models.py
    def __init__(self, F=64, n_int=3, n_rbf=32, rc=5.0, gamma=10.0):
        super().__init__()
        self.emb = nn.Embedding(3, F)
        self.register_buffer("mu", torch.linspace(0, rc, n_rbf)); self.gamma, self.rc = gamma, rc
        self.filt = nn.ModuleList([nn.Sequential(nn.Linear(n_rbf, F), SSP(), nn.Linear(F, F)) for _ in range(n_int)])
        self.in2f = nn.ModuleList([nn.Linear(F, F, bias=False) for _ in range(n_int)])
        self.f2out = nn.ModuleList([nn.Sequential(nn.Linear(F, F), SSP(), nn.Linear(F, F)) for _ in range(n_int)])
        self.out = nn.Sequential(nn.Linear(F, F // 2), SSP(), nn.Linear(F // 2, 1))

    def forward(self, R, species, src, dst, batch):
        B = len(R); pos = R.reshape(-1, 3)
        r = (pos[dst] - pos[src]).norm(dim=-1)
        rbf = torch.exp(-self.gamma * (r[:, None] - self.mu) ** 2)
        fcut = 0.5 * (torch.cos(math.pi * r / self.rc) + 1) * (r < self.rc)
        h = self.emb(species)
        for filt, a, b in zip(self.filt, self.in2f, self.f2out):
            W = filt(rbf) * fcut[:, None]; x = a(h)
            m = torch.zeros_like(x).index_add_(0, dst, x[src] * W)
            h = h + b(m)
        e = self.out(h).squeeze(-1)
        return torch.zeros(B, dtype=e.dtype).index_add_(0, batch, e)


def full_graph(n, B):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij"); m = i != j
    src, dst = i[m], j[m]; off = (torch.arange(B) * n)[:, None]
    return (src[None] + off).reshape(-1), (dst[None] + off).reshape(-1), torch.arange(B).repeat_interleave(n)


class Model(nn.Module):
    def __init__(self, Z, mu, sigma):
        super().__init__(); self.core = SchNetLike(); self._g = {}
        self.register_buffer("species", torch.as_tensor([SPECIES[z] for z in Z]))
        self.mu, self.sigma = float(mu), float(sigma)
    def forward(self, R):
        B = len(R)
        if B not in self._g: self._g[B] = full_graph(NAT, B)
        return self.mu + self.sigma * self.core(R, self.species.repeat(B), *self._g[B])


def energy_forces(model, R, create_graph=False):
    R = R.clone().requires_grad_(True); E = model(R)
    return E, -torch.autograd.grad(E.sum(), R, create_graph=create_graph)[0]


def evaluate(model, R, E, F, bs=100):
    Ep, Fp = [], []
    for i in range(0, len(R), bs):
        e, f = energy_forces(model, R[i:i + bs]); Ep.append(e.detach()); Fp.append(f.detach())
    Ep, Fp = torch.cat(Ep), torch.cat(Fp)
    return (Ep - E).abs().mean().item(), (Fp - F).abs().mean().item()


R3, E3, F3, Z = load(300); R6, E6, F6, _ = load(600); R10, E10, F10, _ = load(1000)
perm = np.random.default_rng(0).permutation(2000)          # as Ch.7 Lab 7.21
tr = torch.as_tensor(perm[:300]); val, test = slice(2000, 2100), slice(2200, 3000)
i6 = np.sort(np.random.default_rng(1).choice(3000, 500, replace=False))
i10 = np.sort(np.random.default_rng(2).choice(1000, 500, replace=False))
Rtr, Etr, Ftr = R3[tr], E3[tr], F3[tr]


def loss_fn(model, R, E, F):
    Ep, Fp = energy_forces(model, R, create_graph=True)
    return (Ep - E).pow(2).mean() + (Fp - F).pow(2).mean()


def train(method, seed, rho=0.05, steps=1200, bs=8, lr=1e-2):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = Model(Z, Etr.mean().item(), Ftr.pow(2).mean().sqrt().item())
    opt = S.SAM(model.parameters(), torch.optim.Adam, rho=rho, mode=method, lr=lr) if method != "adam" \
        else torch.optim.Adam(model.parameters(), lr=lr)
    base = opt.base if method != "adam" else opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, steps, eta_min=lr * 0.01)
    for s in range(steps):
        idx = torch.randint(0, len(Etr), (bs,), generator=gen)
        R, E, F = Rtr[idx], Etr[idx], Ftr[idx]
        def closure():
            opt.zero_grad(); L = loss_fn(model, R, E, F); L.backward(); return L
        if method == "adam": closure(); opt.step()
        else: opt.step(closure)
        sched.step()
    return model


def lam(model, n=64):
    idx = torch.as_tensor(perm[:n])                        # first 64 training frames
    md = copy.deepcopy(model).double()                     # float64 copy for a converged eigsh estimate
    R, E, F = R3[idx].double(), E3[idx].double(), F3[idx].double()
    return S.lambda_max(lambda: loss_fn(md, R, E, F), list(md.parameters()))[0]


if __name__ == "__main__":
    res = {}
    # rho for SAM chosen on the validation frames (seed 0); ASAM uses 10x (its neighbourhood is relative)
    sel = {}
    for r in (0.01, 0.05, 0.2):
        m = train("sam", 0, rho=r); sel[r] = evaluate(m, R3[val], E3[val], F3[val])[1]
        print(f"  SAM rho={r:g}: val force MAE {1e3 * sel[r]:.0f} meV/A  ({time.time() - t0:.0f} s)", flush=True)
    rho = min(sel, key=sel.get)
    for name, m, r in (("Adam", "adam", 0.0), ("SAM", "sam", rho), ("ASAM", "asam", 10 * rho)):
        rows = []
        for seed in (0, 1, 2):
            c0 = time.process_time(); mod = train(m, seed, rho=r); cpu = time.process_time() - c0
            e3, f3 = evaluate(mod, R3[test], E3[test], F3[test]); _, f6 = evaluate(mod, R6[i6], E6[i6], F6[i6])
            _, f10 = evaluate(mod, R10[i10], E10[i10], F10[i10])
            rows.append(dict(E300=e3, F300=f3, F600=f6, F1000=f10, lam=lam(mod), cpu=cpu))
            torch.save(mod.state_dict(), os.path.join(HERE, f"83_mlip_{m}_s{seed}.pt"))
            print("   %-5s seed %d: E300 %5.1f meV | F300 %4.0f F600 %4.0f F1000 %4.0f meV/A | lam_max %8.1f | cpu %4.0f s" % (
                name, seed, 1e3 * e3, 1e3 * f3, 1e3 * f6, 1e3 * f10, rows[-1]["lam"], cpu), flush=True)
        res[name] = rows
        A = {k: np.array([x[k] for x in rows]) for k in rows[0]}
        print("%-5s (rho %g) F300 %4.0f+/-%2.0f  F600 %4.0f+/-%2.0f  F1000 %4.0f+/-%2.0f meV/A  ratio 1000/300 %.2f  lam_max %.0f+/-%.0f" % (
            name, r, 1e3 * A["F300"].mean(), 1e3 * A["F300"].std(), 1e3 * A["F600"].mean(), 1e3 * A["F600"].std(),
            1e3 * A["F1000"].mean(), 1e3 * A["F1000"].std(), (A["F1000"] / A["F300"]).mean(), A["lam"].mean(), A["lam"].std()), flush=True)
    json.dump(dict(res=res, rho=rho, sel={str(k): v for k, v in sel.items()}), open(os.path.join(HERE, "83_mlip_results.json"), "w"))
    print("wall %.0f s" % (time.time() - t0))
