"""Lab 9.3-2: DeepONet vs FNO on the Burgers operator u0 -> u(.,T) (same data as 93_fno_burgers.py, 64 sensors).
(1) POD spectrum of the outputs = floor for ANY linear decoder with p basis functions
(2) DeepONet / POD-DeepONet / FNO test error versus the number of training functions
(3) the same in the co-moving (Galilean) frame, where the mean advection is removed by symmetry
"""
import os, sys, time, json, math, importlib
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models")
t_all = time.time(); c_all = time.process_time()

d = np.load(os.path.join(HERE, "..", "data", "93_burgers.npz"))
traj = torch.tensor(d["traj"]); NF, T, S = traj.shape[-1], float(d["T"]), 64
A, U = traj[:, 0], traj[:, -1]
te = slice(2000, 2200)


def comoving(A, U):
    """Galilean frame: v0 = u0 - mean, v(y,T) = u(y + mean*T, T) - mean (exact Fourier shift on the fine grid)."""
    m = A.mean(1, keepdim=True)
    k = 2 * math.pi * torch.arange(NF // 2 + 1)
    V = torch.fft.irfft(torch.fft.rfft(U) * torch.exp(1j * k * m * T), n=NF) - m
    return A - m, V


frames = {"lab": (A[:, ::NF // S], U[:, ::NF // S])}
A0, V = comoving(A, U); frames["comoving"] = (A0[:, ::NF // S], V[:, ::NF // S])
y = (torch.arange(S) / S)[:, None]
out = {"p": [2, 4, 8, 16, 32, 64]}

# ---------------- (1) POD floor ----------------
for fr, (a, u) in frames.items():
    mean = u[:1000].mean(0); sv = torch.linalg.svd(u[:1000] - mean, full_matrices=False)
    floor = []
    for p in out["p"]:
        B = sv.Vh[:p]; r = (u[te] - mean) - (u[te] - mean) @ B.T @ B
        floor.append((r.norm(dim=1) / u[te].norm(dim=1)).mean().item())
    out[f"pod_floor_{fr}"] = floor; out[f"sv_{fr}"] = (sv.S / sv.S[0]).tolist()
    print(f"[{fr:8s}] POD floor (test rel-L2 of the best p-term linear reconstruction): " + "  ".join(f"p={p}: {x:.4f}" for p, x in zip(out['p'], floor)))
rel_fluct = (frames["comoving"][1][te].norm(dim=1) / frames["lab"][1][te].norm(dim=1)).mean().item()
print(f"norm ratio ||v(T)|| / ||u(T)|| (test mean) = {rel_fluct:.3f}  (errors in the two frames are relative to different norms)")
out["norm_ratio"] = rel_fluct


# ---------------- (2), (3) learning curves ----------------
def run_don(a, u, ntr, p=64, pod=False, updates=6000, seed=0):
    torch.manual_seed(seed)
    if pod:
        mean = u[:ntr].mean(0); Vh = torch.linalg.svd(u[:ntr] - mean, full_matrices=False).Vh[:p]
        net = M.DeepONet(S, p=p, basis=(Vh.T * math.sqrt(S)).contiguous(), mean=mean)
    else:
        net = M.DeepONet(S, p=p, width=128, depth=3, periodic=4)
    bs = 50; ep = max(1, round(updates * bs / ntr))
    h, wall = M.fit(net, lambda idx: M.rel_l2(net(a[idx], y), u[idx]), ntr, ep, bs, lr=2e-3, seed=seed)
    with torch.no_grad(): e = M.rel_l2(net(a[te], y), u[te]).item()
    return e, h[-1], wall, M.count_params(net)


def run_fno(a, u, ntr, updates=2500, seed=0):
    torch.manual_seed(seed)
    net = M.FNO1d(width=32, kmax=16, depth=4); bs = 20; ep = max(1, round(updates * bs / ntr))
    h, wall = M.fit(net, lambda idx: M.rel_l2(net(a[idx, None]), u[idx, None]), ntr, ep, bs, lr=3e-3, seed=seed)
    with torch.no_grad(): e = M.rel_l2(net(a[te, None]), u[te, None]).item()
    return e, h[-1], wall, M.count_params(net)


out["ntr"] = [250, 500, 1000, 2000]
a, u = frames["lab"]
for name, fn in (("deeponet", lambda n: run_don(a, u, n)), ("pod_deeponet", lambda n: run_don(a, u, n, p=32, pod=True)),
                 ("fno", lambda n: run_fno(a, u, n))):
    res = [fn(n) for n in out["ntr"]]
    out[name] = [r[0] for r in res]; out[name + "_train"] = [r[1] for r in res]; out[name + "_wall"] = [r[2] for r in res]
    print(f"[lab] {name:13s} ({res[0][3]:6d} params): test rel-L2 " + "  ".join(f"N={n}: {r[0]:.4f} (train {r[1]:.4f})" for n, r in zip(out["ntr"], res))
          + f"   [{sum(r[2] for r in res):.0f} s]")
a, u = frames["comoving"]
for name, fn in (("deeponet", lambda n: run_don(a, u, n)), ("pod_deeponet", lambda n: run_don(a, u, n, p=32, pod=True)),
                 ("fno", lambda n: run_fno(a, u, n))):
    e, tr_, wall, _ = fn(1000); out[name + "_comoving"] = e
    print(f"[comoving, N=1000] {name:13s}: test rel-L2 {e:.4f} (train {tr_:.4f})  -> relative to ||u(T)||: {e*rel_fluct:.4f}   [{wall:.0f} s]")

# ---------------- translation test: DeepONet has no built-in equivariance ----------------
a, u = frames["lab"]
torch.manual_seed(0); net = M.DeepONet(S, p=64, width=128, depth=3, periodic=4)
M.fit(net, lambda idx: M.rel_l2(net(a[idx], y), u[idx]), 1000, 300, 50, lr=2e-3, seed=0)
with torch.no_grad():
    eq = M.rel_l2(net(torch.roll(a[te], 5, -1), y), torch.roll(net(a[te], y), 5, -1)).item()
    out["equiv_deeponet"] = eq
    j = int(torch.argsort(M.rel_l2_each(net(a[te], y), u[te]))[100])
    out["example"] = dict(a=a[te][j].tolist(), u=u[te][j].tolist(), don=net(a[te], y)[j].tolist())
print(f"DeepONet translation-equivariance error (shift 5/64): {eq:.3f}")
json.dump(out, open(os.path.join(HERE, "93_deeponet.json"), "w"))
print(f"wall {time.time()-t_all:.0f} s (CPU {time.process_time()-c_all:.0f} s)")
