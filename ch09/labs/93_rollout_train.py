"""Lab 9.3-3 (training): one-step FNO surrogates for Kuramoto-Sivashinsky, u(t) -> u(t + 1).
All variants share the same one-step pre-training and are then fine-tuned for the SAME number of updates:
  onestep   : continue one-step training (control)
  noise     : one-step training with Gaussian noise added to the input
  push      : pushforward - input is the model's own (detached) prediction after 1-2 steps
  unroll    : 4 unrolled steps with back-propagation through all of them
  cons      : as `onestep`, but the zero Fourier mode of the increment is removed in the architecture
Data: data/93_ks.npz (L = 64, N = 128, solver dt = 0.025, stored every 1.0).
"""
import os, sys, time, json, importlib, copy
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M = importlib.import_module("93_models")
t_all = time.time(); c_all = time.process_time()

d = np.load(os.path.join(HERE, "..", "data", "93_ks.npz"))
traj = torch.tensor(d["traj"]); NTR, NS = int(d["ntr"]), int(d["ns_tr"])        # (80, 401, 128)
X = traj[:NTR, :NS + 1]                                                        # training trajectories, 200 steps
KMAX, WIDTH, BATCH = 24, 32, 64
PRE_EP, FT_UPD, KU = 6, 600, 4
make = lambda cons=None: M.FNO1d(width=WIDTH, kmax=KMAX, depth=4, residual=True, conserve=cons, L=64.0)


def batch(g, K):
    i = torch.randint(0, NTR, (BATCH,), generator=g); n = torch.randint(0, NS - K + 1, (BATCH,), generator=g)
    return [X[i, n + j][:, None] for j in range(K + 1)]                        # u(t), u(t+1), ..., u(t+K)


def loss_fn(net, mode, g):
    if mode in ("onestep", "cons"):
        u = batch(g, 1); return M.rel_l2(net(u[0]), u[1])
    if mode == "noise":
        u = batch(g, 1); return M.rel_l2(net(u[0] + SIGMA * torch.randn(u[0].shape, generator=g)), u[1])
    if mode == "push":
        k = int(torch.randint(0, 3, (1,), generator=g)); u = batch(g, k + 1); v = u[0]
        with torch.no_grad():
            for _ in range(k): v = net(v)                                      # the model's own inputs, no gradient
        return M.rel_l2(net(v), u[k + 1])
    if mode == "unroll":
        u = batch(g, KU); v = u[0]; loss = 0.0
        for j in range(KU):
            v = net(v); loss = loss + M.rel_l2(v, u[j + 1]) / KU               # gradient through all steps
        return loss


def run(net, mode, updates, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=updates, pct_start=0.1)
    c0 = time.process_time(); t0 = time.time()
    for it in range(updates):
        loss = loss_fn(net, mode, g); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return time.time() - t0, time.process_time() - c0


@torch.no_grad()
def onestep_err(net):
    te = traj[NTR:]
    return M.rel_l2(net(te[:, :-1].reshape(-1, 1, 128)), te[:, 1:].reshape(-1, 1, 128)).item()


pre_upd = PRE_EP * (NTR * NS // BATCH)
out = {}; sd = {}
print(f"KS one-step data: {NTR} trajectories x {NS} steps = {NTR*NS} pairs; FNO width {WIDTH}, kmax {KMAX}, "
      f"{M.count_params(make())} parameters; pre-training {pre_upd} updates, fine-tuning {FT_UPD} updates, batch {BATCH}")
for name in ("base", "cons"):
    torch.manual_seed(0); net = make("mean" if name == "cons" else None)
    w, c = run(net, "onestep", pre_upd, 2e-3, seed=1)
    out[f"pre_{name}"] = dict(err1=onestep_err(net), wall=w, cpu=c); sd[f"pre_{name}"] = copy.deepcopy(net.state_dict())
    print(f"pre-train [{name:4s}]: one-step test rel-L2 {out[f'pre_{name}']['err1']:.4f}   ({w:.0f} s wall, {c:.0f} s CPU)")
SIGMA = 0.5 * out["pre_base"]["err1"] * float(X.std())                         # noise ~ half the one-step RMS error
print(f"noise level sigma = {SIGMA:.4f} (rms u = {float(X.std()):.3f})")
for name in ("onestep", "noise", "push", "unroll", "cons"):
    net = make("mean" if name == "cons" else None); net.load_state_dict(sd["pre_cons" if name == "cons" else "pre_base"])
    w, c = run(net, name, FT_UPD, 5e-4, seed=2)
    out[name] = dict(err1=onestep_err(net), wall=w, cpu=c); sd[name] = copy.deepcopy(net.state_dict())
    print(f"fine-tune [{name:7s}]: one-step test rel-L2 {out[name]['err1']:.4f}   ({w:.0f} s wall, {c:.0f} s CPU)")
out["sigma"] = SIGMA
torch.save(sd, os.path.join(HERE, "93_rollout_models.pt")); json.dump(out, open(os.path.join(HERE, "93_rollout_train.json"), "w"))
print(f"wall {time.time()-t_all:.0f} s (CPU {time.process_time()-c_all:.0f} s)")
