"""Ex 7.6.6: why AdamW differs more than SGD between DDP and single-process training (Lab 7.20 setup).
DDP is emulated exactly in one process: rank r of DistributedSampler gets perm[r::2], so global batch k is
perm[256k:256(k+1)] split into its even/odd entries; DDP's gradient = (mean grad over half 0 + mean grad over half 1)/2."""
import os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
src = open("../ch07/labs/76_ddp.py").read().split('if __name__ == "__main__":')[0]
src = src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", "HERE = '../ch07/labs'")
exec(src)
X, y = load()
def opt_for(params, name, eps):
    if name == "adamw": return torch.optim.AdamW(params, lr=LR, eps=eps)
    return torch.optim.SGD(params, lr=0.05, momentum=0.9)
def run(H, name, mode, eps=1e-8, max_steps=None, return_grad=False):
    m = make_model(H); opt = opt_for(m.parameters(), name, eps); step = 0
    for ep in range(EPOCHS):
        g = torch.Generator(); g.manual_seed(SEED + ep); perm = torch.randperm(len(y), generator=g)
        for s in range(0, len(y), B_GLOBAL):
            idx = perm[s:s + B_GLOBAL]; opt.zero_grad()
            if mode == "single":
                loss_fn(m, X[idx], y[idx]).backward()
            else:                                         # DDP emulation: average of two half-batch gradients
                grads = []
                for r in range(2):
                    m.zero_grad(); loss_fn(m, X[idx[r::2]], y[idx[r::2]]).backward()
                    grads.append([p.grad.clone() for p in m.parameters()])
                for p, a, b in zip(m.parameters(), *grads): p.grad = (a / 2 + b / 2)
            if return_grad and step == 0:
                G = [p.grad.clone() for p in m.parameters()]
            opt.step(); step += 1
            if max_steps and step >= max_steps:
                return (m, G) if return_grad else m
    return m
def dmax(a, b): return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
H = 128
print("--- full training (4 epochs, 68 steps), emulated DDP vs single process; compare Lab 7.20: AdamW 4.9e-07, SGD 1.0e-07")
for name in ("adamw", "sgd"):
    print(f"{name}: final max|dtheta| = {dmax(run(H, name, 'single'), run(H, name, 'ddp')):.1e}")
print("--- (a) after the FIRST update")
for name in ("adamw", "sgd"):
    ms, Gs = run(H, name, "single", max_steps=1, return_grad=True); md, Gd = run(H, name, "ddp", max_steps=1, return_grad=True)
    dg = max((a - b).abs().max().item() for a, b in zip(Gs, Gd))
    print(f"{name}: max|dg| = {dg:.1e}, max|dtheta| after step 1 = {dmax(ms, md):.1e}")
    if name == "adamw":
        g = torch.cat([x.flatten() for x in Gs]); dgv = torch.cat([(a - b).flatten() for a, b in zip(Gs, Gd)])
        th_s = torch.cat([p.detach().flatten() for p in ms.parameters()]); th_d = torch.cat([p.detach().flatten() for p in md.parameters()])
        dth = (th_s - th_d).abs()
        k = dth.argmax(); print(f"   largest dtheta at |g| = {g[k].abs():.1e}, dg = {dgv[k]:.1e}; components with |g| < 1e-6: {(g.abs() < 1e-6).sum().item()} of {g.numel()}, exactly zero: {(g == 0).sum().item()}")
        for lo, hi in ((0, 1e-9), (1e-9, 1e-7), (1e-7, 1e-5), (1e-5, 1)):
            sel = (g.abs() >= lo) & (g.abs() < hi) & (dgv != 0)
            if sel.any():
                amp = (dth[sel] / dgv[sel].abs()).median().item()
                print(f"   |g| in [{lo:.0e},{hi:.0e}): n={sel.sum().item():6d} with dg!=0, median |dtheta|/|dg| = {amp:.2e}  (theory eta*eps/(|g|+eps)^2 at median |g| = {(1e-3*1e-8/(g[sel].abs().median()+1e-8)**2).item():.2e})")
print("--- (b) eps sweep, AdamW: max|dtheta| after step 1 and after 68 steps")
for eps in (1e-8, 1e-6, 1e-4, 1e-3):
    a1 = dmax(run(H, "adamw", "single", eps, max_steps=1), run(H, "adamw", "ddp", eps, max_steps=1))
    aF = dmax(run(H, "adamw", "single", eps), run(H, "adamw", "ddp", eps))
    print(f"eps {eps:.0e}: step 1 {a1:.1e} | final {aF:.1e}")
