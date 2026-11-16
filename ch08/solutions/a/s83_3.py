"""Ex 8.3.3: AdamW, batch 32/64/256/1024 at the SAME number of steps (2500), linear lr scaling; 3 seeds.
Also Ex 8.3.2(c): block-restricted lambda_max of the AdamW seed-0 model."""
import os, sys, time, json, numpy as np, torch
torch.set_num_threads(1)
HERE = "../ch08/labs"
src = open(os.path.join(HERE, "83_bandgap_sam.py")).read().split("sub = np.random.default_rng(1)")[0]
src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))', f'HERE = {HERE!r}')
g = {"__file__": os.path.join(HERE, "83_bandgap_sam.py"), "__name__": "x"}; exec(compile(src, "83_bandgap_sam.py", "exec"), g)
Z, Y, make_mlp, sharpness, S, mae = g["Z"], g["Y"], g["make_mlp"], g["sharpness"], g["S"], g["mae"]
sp = np.load(os.path.join(HERE, "83_bandgap_split.npz")); tr_all, te, ox, sub = sp["tr_all"], sp["te"], sp["ox"], sp["sub"]
t0 = time.time()

# ---- 8.3.2(c): which blocks carry the curvature? -------------------------------------------------
m = make_mlp(Z(te[:1]).shape[1]); m.load_state_dict(torch.load(os.path.join(HERE, "83_bg_adamw_s0.pt")))
m = m.double(); Xs, ys = Z(sub).double(), Y(sub).double()          # float64 for converged eigsh estimates
L = lambda: (m(Xs).squeeze(-1) - ys).pow(2).mean()
blocks = {"all": list(m.parameters()), "W1,b1": [m[0].weight, m[0].bias], "W2": [m[2].weight],
          "b2,W3,b3": [m[2].bias, m[4].weight, m[4].bias]}
for k, ps in blocks.items():
    print(f"8.3.2(c) lambda_max restricted to {k:9s}: {S.lambda_max(L, ps)[0]:9.3f}")
# full top eigenvector (eigsh, unit norm): weight in each block
ps = list(m.parameters()); v = S.lambda_max(L, ps, return_vec=True)[2]
w = [float((x**2).sum()) for x in v]
print("8.3.2(c) top eigvec weight: W1,b1 %.3f  W2 %.3f  b2,W3,b3 %.3f" % (w[0] + w[1], w[2], w[3] + w[4] + w[5]))
for a in (4.0, 0.25):
    D = [1 / a, 1 / a, a, 1, 1, 1]
    # Rayleigh quotient of the transformed Hessian A^-1 H A^-1 along the transformed old top vector
    print(f"   a={a}: predicted by block scaling: 16*lam(W2)={16*S.lambda_max(L, blocks['W2'])[0]:.1f}" if a == 4 else
          f"   a={a}: 16*lam(W1,b1)={16*S.lambda_max(L, blocks['W1,b1'])[0]:.1f}")
print(f"[{time.time()-t0:.0f} s]", flush=True)

# ---- 8.3.3 -------------------------------------------------------------------------------------------
Xtr, ytr = Z(tr_all), Y(tr_all); n = len(ytr)
def fit_steps(B, seed, steps=2500, lr=None, wd=1e-4):
    lr = lr or 1e-3 * B / 64
    model = make_mlp(Xtr.shape[1], seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed); idx = torch.randperm(n, generator=gen); pos = 0
    for st in range(steps):
        if pos + B > n: idx = torch.randperm(n, generator=gen); pos = 0
        j = idx[pos:pos + B]; pos += B
        loss = (model(Xtr[j]).squeeze(-1) - ytr[j]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return model
@torch.no_grad()
def train_mse(model): return (model(Xtr).squeeze(-1) - ytr).pow(2).mean().item()
out = {}
for B in (32, 64, 256, 1024):
    rows = []
    for seed in (0, 1, 2):
        c0 = time.process_time(); mod = fit_steps(B, seed)
        rows.append(dict(mse=train_mse(mod), trmae=mae(mod, tr_all), id=mae(mod, te), ox=mae(mod, ox), lam=sharpness(mod, sub),
                         lamT=sharpness(mod, sub, adaptive=True), cpu=time.process_time() - c0))
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}; out[B] = rows
    print(f"B={B:5d} lr={1e-3*B/64:.1e} epochs={2500*B/n:6.1f} | train MSE {A['mse'].mean():.4f}+/-{A['mse'].std():.4f} (MAE {A['trmae'].mean():.3f}) | "
          f"lam_max {A['lam'].mean():6.1f}+/-{A['lam'].std():4.1f} | lam(THT) {A['lamT'].mean():.3f} | ID {A['id'].mean():.3f}+/-{A['id'].std():.3f} | "
          f"oxide {A['ox'].mean():.3f}+/-{A['ox'].std():.3f} | cpu {A['cpu'].mean():.0f} s", flush=True)
# control: B=1024 with the lab's lr (4e-3) at 2500 steps
rows = []
for seed in (0, 1, 2):
    mod = fit_steps(1024, seed, lr=4e-3)
    rows.append(dict(mse=train_mse(mod), id=mae(mod, te), ox=mae(mod, ox), lam=sharpness(mod, sub)))
A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
print(f"B= 1024 lr=4.0e-03 (control) | train MSE {A['mse'].mean():.4f} | lam_max {A['lam'].mean():.1f}+/-{A['lam'].std():.1f} | ID {A['id'].mean():.3f} | oxide {A['ox'].mean():.3f}")
json.dump({str(k): v for k, v in out.items()}, open("s83_3.json", "w"))
print(f"wall {time.time()-t0:.0f} s")
