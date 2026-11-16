"""Lab 7.1.3 -- optimizers on a materials regression (Magpie -> experimental band gap).
A: SGD / momentum / Nesterov / Adagrad / RMSProp / Adam / AdamW, LR chosen on a validation split.
B: Adam+L2 vs AdamW -- (i) one-parameter toy, (ii) band-gap MLP over a lambda grid.
C: warmup+cosine schedule, L-BFGS (full batch), bf16 autocast on CPU, fp16 underflow & loss scaling.
Same 5-fold split as Ch.6 Lab 6.13 (KFold shuffle seed 0); here fold 0 only.
Run: python 71_optim.py   (about 6-8 min on one shared CPU core)"""
import time, math, numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(1)
T0 = time.time()
df = pd.read_parquet("../data/magpie_expt_gap.parquet")
X = df.drop(columns=["formula", "target"]).values.astype(np.float64); y = df.target.values
tr, te = next(iter(KFold(5, shuffle=True, random_state=0).split(X)))          # Ch.6 fold 0
tr_in, va = train_test_split(tr, test_size=0.1, random_state=0)               # inner validation for LR choice

def prep(fit_idx, *idx):
    vt = VarianceThreshold(1e-6).fit(X[fit_idx]); sc = StandardScaler().fit(vt.transform(X[fit_idx]))
    f = lambda i: torch.tensor(np.clip(sc.transform(vt.transform(X[i])), -5, 5), dtype=torch.float32)
    return [f(i) for i in (fit_idx,) + idx]
def T(v): return torch.tensor(v, dtype=torch.float32)
Xa, Xv, Xt = prep(tr_in, va, te)
ya, yv, yt = T(y[tr_in]), T(y[va]), T(y[te])
print(f"n={len(y)}, p={X.shape[1]} -> {Xa.shape[1]} after variance filter; fold-0 train {len(tr_in)} / val {len(va)} / test {len(te)}")

def mlp(seed, p=Xa.shape[1]):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(p, 256), torch.nn.SiLU(), torch.nn.Linear(256, 128),
                               torch.nn.SiLU(), torch.nn.Linear(128, 1))
def mae(net, Xe, ye):
    with torch.no_grad(): return (net(Xe)[:, 0].clamp(min=0) - ye).abs().mean().item()

def fit(make_opt, seed=0, epochs=80, bs=64, sched=None, Xtr=Xa, ytr=ya, Xev=(Xv,), yev=(yv,), autocast=False):
    net = mlp(seed); opt = make_opt(net.parameters())
    nb = math.ceil(len(ytr) / bs); total = epochs * nb
    lr0 = [g["lr"] for g in opt.param_groups]; step = 0; g = torch.Generator().manual_seed(seed)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(len(ytr), generator=g)
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            if sched is not None:
                for gr, l0 in zip(opt.param_groups, lr0): gr["lr"] = l0 * sched(step, total)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out = net(Xtr[idx])[:, 0]
            loss = ((out.float() - ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        curve.append([mae(net, Xe, ye) for Xe, ye in zip(Xev, yev)])
    return net, np.array(curve)

O = torch.optim
opts = {"SGD":           (lambda lr: (lambda p: O.SGD(p, lr=lr)),                               [3e-3, 1e-2, 3e-2]),
        "SGD+momentum":  (lambda lr: (lambda p: O.SGD(p, lr=lr, momentum=0.9)),                 [1e-3, 3e-3, 1e-2]),
        "Nesterov":      (lambda lr: (lambda p: O.SGD(p, lr=lr, momentum=0.9, nesterov=True)),  [1e-3, 3e-3, 1e-2]),
        "Adagrad":       (lambda lr: (lambda p: O.Adagrad(p, lr=lr)),                           [3e-3, 1e-2, 3e-2]),
        "RMSProp":       (lambda lr: (lambda p: O.RMSprop(p, lr=lr)),                           [1e-4, 3e-4, 1e-3]),
        "Adam":          (lambda lr: (lambda p: O.Adam(p, lr=lr)),                              [3e-4, 1e-3, 3e-3]),
        "AdamW(wd=1e-2)": (lambda lr: (lambda p: O.AdamW(p, lr=lr, weight_decay=1e-2)),         [3e-4, 1e-3, 3e-3])}
print("=== Part A: optimizers (80 epochs, batch 64, constant LR; LR picked by min val MAE; seeds 0-2) ===")
print(f"{'optimizer':15s} {'best LR':>8s} {'val MAE':>8s} {'test MAE (mean+/-sd, 3 seeds)':>30s} {'ep to val<0.5':>14s}")
curves = {}
for name, (mk, lrs) in opts.items():
    best = None
    for lr in lrs:
        _, c = fit(mk(lr), seed=0)
        v = c[:, 0].min()
        if best is None or v < best[1]: best = (lr, v, c[:, 0])
    lr = best[0]; tests = []
    for s in range(3):
        net, c = fit(mk(lr), seed=s, Xtr=torch.cat([Xa, Xv]), ytr=torch.cat([ya, yv]), Xev=(Xt,), yev=(yt,))
        tests.append(c[-1, 0])
        if s == 0: curves[name] = c[:, 0]
    reach = np.argmax(best[2] < 0.5) + 1 if (best[2] < 0.5).any() else -1
    print(f"{name:15s} {lr:8.0e} {best[1]:8.3f} {np.mean(tests):22.3f} +/- {np.std(tests):.3f} {reach:14d}")
print(f"[time so far {time.time()-T0:.0f} s]")

print("=== Part B(i): one parameter, zero-mean gradient noise of scale sigma; theta0=1, lr=1e-3, lambda=0.5, 2000 steps ===")
def toy(kind, sigma, steps=2000, lr=1e-3, lam=0.5, seed=0):
    th = torch.tensor([1.0], requires_grad=True); g = torch.Generator().manual_seed(seed)
    opt = O.Adam([th], lr=lr, weight_decay=lam) if kind == "Adam+L2" else O.AdamW([th], lr=lr, weight_decay=lam)
    for _ in range(steps):
        opt.zero_grad(); th.grad = sigma * torch.randn(1, generator=g); opt.step()   # data gradient only
    return th.item()
print(f"{'sigma':>8s} {'Adam+L2 |theta|':>16s} {'AdamW |theta|':>14s}   (AdamW decay alone: exp(-lr*lam*steps) = {math.exp(-1e-3*0.5*2000):.3f})")
toyres = []
for sg in [0.0, 1e-3, 1e-1, 1.0, 10.0]:
    a_ = np.mean([abs(toy("Adam+L2", sg, seed=s)) for s in range(20)])
    w_ = np.mean([abs(toy("AdamW", sg, seed=s)) for s in range(20)])
    toyres.append((sg, a_, w_)); print(f"{sg:8.0e} {a_:16.3f} {w_:14.3f}")

print("=== Part B(ii): band-gap MLP, Adam(lr 1e-3) with coupled L2 vs AdamW (decoupled), test MAE (seed 0) ===")
lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
Xf, yf = torch.cat([Xa, Xv]), torch.cat([ya, yv])
wdres = {"Adam+L2": [], "AdamW": []}
for lam in lams:
    row = []
    for kind in wdres:
        mk = (lambda p, lam=lam: O.Adam(p, lr=1e-3, weight_decay=lam)) if kind == "Adam+L2" else \
             (lambda p, lam=lam: O.AdamW(p, lr=1e-3, weight_decay=lam))
        net, c = fit(mk, seed=0, Xtr=Xf, ytr=yf, Xev=(Xt,), yev=(yt,))
        wn = sum((p ** 2).sum() for n_, p in net.named_parameters() if "weight" in n_).sqrt().item()
        wdres[kind].append((c[-1, 0], wn)); row.append(f"{kind}: MAE {c[-1,0]:.3f}  ||W|| {wn:6.1f}")
    print(f"lambda {lam:7.0e}   " + "   |   ".join(row))
print(f"[time so far {time.time()-T0:.0f} s]")

print("=== Part C: schedules, L-BFGS, bf16 autocast (train = fold-0 train, test = fold-0 test, seed 0) ===")
warm = lambda s, tot: min(1.0, (s + 1) / (0.05 * tot)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tot)))
t1 = time.time(); _, cA = fit(lambda p: O.AdamW(p, lr=1e-3, weight_decay=1e-2), Xtr=Xf, ytr=yf, Xev=(Xt,), yev=(yt,)); tA = time.time() - t1
_, cB = fit(lambda p: O.AdamW(p, lr=3e-3, weight_decay=1e-2), sched=warm, Xtr=Xf, ytr=yf, Xev=(Xt,), yev=(yt,))
print(f"AdamW constant lr 1e-3          : final test MAE {cA[-1,0]:.3f}  (min over epochs {cA[:,0].min():.3f}; final-5 epoch std {cA[-5:,0].std():.4f})")
print(f"AdamW warmup(5%)+cosine, peak 3e-3: final test MAE {cB[-1,0]:.3f}  (min over epochs {cB[:,0].min():.3f}; final-5 epoch std {cB[-5:,0].std():.4f})")
t1 = time.time(); _, cC = fit(lambda p: O.AdamW(p, lr=1e-3, weight_decay=1e-2), Xtr=Xf, ytr=yf, Xev=(Xt,), yev=(yt,), autocast=True); tC = time.time() - t1
print(f"AdamW constant lr 1e-3, bf16 autocast: final test MAE {cC[-1,0]:.3f};  wall {tC:.1f} s vs fp32 {tA:.1f} s")
net = mlp(0); lb = O.LBFGS(net.parameters(), lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
def closure():
    lb.zero_grad(); l = ((net(Xf)[:, 0] - yf) ** 2).mean(); l.backward(); return l
t1 = time.time(); lcur = []
for it in range(40):
    lb.step(closure); lcur.append(mae(net, Xt, yt))
print(f"L-BFGS full batch (40 x 20 iterations): test MAE after 200 it {lcur[9]:.3f}, after 800 it {lcur[-1]:.3f}; "
      f"min {min(lcur):.3f}; wall {time.time()-t1:.1f} s")
with torch.no_grad():
    trmae = (net(Xf)[:, 0].clamp(min=0) - yf).abs().mean().item()
print(f"   L-BFGS train MAE {trmae:.3f}  (full-batch second-order-like optimizer, no noise, no regularization)")
h = torch.tensor([1e-8, 1e-5, 3e-4])
print("fp16 underflow: ", h.half().tolist(), " | scaled by 2^16 then cast: ", (h * 65536).half().tolist(),
      " | unscaled back in fp32: ", [f"{v:.3e}" for v in ((h * 65536).half().float() / 65536).tolist()])
print("bf16 keeps range:", [f"{v:.3e}" for v in h.bfloat16().float().tolist()])

import lightgbm as lgb
m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=0, n_jobs=1, verbose=-1)
m.fit(X[tr], y[tr]); p_ = m.predict(X[te])
print(f"reference: LightGBM (Lab 6.13 settings) on the same fold 0: test MAE {np.abs(p_-y[te]).mean():.3f}, clipped at 0: {np.abs(np.clip(p_,0,None)-y[te]).mean():.3f}")
np.savez("71_optim.npz", names=np.array(list(curves)), curves=np.array(list(curves.values())),
         toy=np.array(toyres), lams=lams, wd_adam=np.array(wdres["Adam+L2"]), wd_adamw=np.array(wdres["AdamW"]),
         cA=cA[:, 0], cB=cB[:, 0], lbfgs=np.array(lcur))
print(f"wall {time.time()-T0:.0f} s")
