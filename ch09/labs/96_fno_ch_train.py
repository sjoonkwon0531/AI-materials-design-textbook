"""Case Study 9.6, training: FNO-2D surrogate of the Cahn-Hilliard solver.
Learns the map  c(t) -> c(t + 2.5)  (50 solver steps of dt = 0.05) from 160 trajectories (3200 pairs).
Usage: python 96_fno_ch_train.py [conserve: 0 | 1] [seed] [loss: one | pf]
   conserve = 0 : plain FNO (increment unconstrained)
   conserve = 1 : zero-mode constraint, the mean of the increment is removed (mass conserved by construction)
   loss = one   : one-step loss  ||G(c_t) - c_{t+1}|| / ||c_{t+1}||
   loss = pf    : pushforward loss, one-step term + ||G(sg[G(c_t)]) - c_{t+2}|| / ||c_{t+2}||  (sg = stop-gradient)
Saves ../data/96_fno_ch_{free|cons}[_pf].pt . One thread."""
import sys, time, importlib.util
import numpy as np
import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
fno = load("fno", "95_fno2d.py")

CONSERVE = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LOSS = sys.argv[3] if len(sys.argv) > 3 else "one"
EPOCHS, BATCH, LR, MODES, WIDTH = 5, 8, 2e-3, 16, 20
torch.set_num_threads(1); torch.manual_seed(SEED); np.random.seed(SEED)
tag = ("cons" if CONSERVE else "free") + ("_pf" if LOSS == "pf" else "")

d = np.load("../data/96_ch_64.npz")
train, test = d["train"], d["test"]                       # (160, 21, 64, 64), (40, 21, 64, 64)
if LOSS == "pf":                                          # triples (c_t, c_{t+1}, c_{t+2})
    x_tr = torch.from_numpy(train[:, :-2].reshape(-1, 64, 64))
    y_tr = torch.from_numpy(train[:, 1:-1].reshape(-1, 64, 64))
    z_tr = torch.from_numpy(train[:, 2:].reshape(-1, 64, 64))
else:                                                     # pairs (c_t, c_{t+1})
    x_tr = torch.from_numpy(train[:, :-1].reshape(-1, 64, 64))
    y_tr = torch.from_numpy(train[:, 1:].reshape(-1, 64, 64))
x_te = torch.from_numpy(test[:, :-1].reshape(-1, 64, 64))
y_te = test[:, 1:]

model = fno.FNO2d(modes=MODES, width=WIDTH, n_layers=4, proj=64, conserve=CONSERVE)
print(f"[{tag}] FNO-2D modes {MODES}, width {WIDTH}, 4 layers, {fno.n_params(model):,d} real parameters; "
      f"{len(x_tr)} training {'triples' if LOSS == 'pf' else 'pairs'}, loss '{LOSS}', seed {SEED}")
steps_total = EPOCHS * (len(x_tr) // BATCH)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps_total, pct_start=0.1)

def one_step(model):
    with torch.no_grad():
        p = torch.cat([model(x_te[i:i + 40]) for i in range(0, len(x_te), 40)]).numpy().reshape(y_te.shape)
    return np.sqrt(((p - y_te) ** 2).sum((2, 3)) / (y_te**2).sum((2, 3)))      # (40, 20)

t0 = time.perf_counter(); c0 = time.process_time(); hist = []
for ep in range(EPOCHS):
    perm = torch.randperm(len(x_tr)); run = 0.0
    for i in range(0, len(perm) - BATCH + 1, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        pred = model(x_tr[idx])
        loss = fno.rel_l2_loss(pred, y_tr[idx])
        if LOSS == "pf":                                  # the model sees its own output; no gradient through step 1
            loss = loss + fno.rel_l2_loss(model(pred.detach()), z_tr[idx])
        loss.backward(); opt.step(); sched.step()
        run += loss.item()
    e = one_step(model)
    hist.append((ep + 1, time.perf_counter() - t0, run / (len(perm) // BATCH), e.mean(), e[:, 0].mean(), e[:, 1:].mean()))
    print(f"[{tag}] epoch {ep + 1}: train loss {hist[-1][2]:.4f} | test one-step rel. L2: all {e.mean():.4f},"
          f" first step {e[:, 0].mean():.4f}, later steps {e[:, 1:].mean():.4f} | {hist[-1][1]:.0f} s", flush=True)
wall, cpu = time.perf_counter() - t0, time.process_time() - c0
print(f"[{tag}] training: wall {wall:.0f} s on a shared machine, CPU time {cpu:.0f} s ({steps_total} Adam steps, batch {BATCH})")
torch.save({"state_dict": model.state_dict(), "hist": np.array(hist), "wall": wall, "cpu": cpu,
            "config": dict(modes=MODES, width=WIDTH, n_layers=4, proj=64, conserve=CONSERVE)},
           f"../data/96_fno_ch_{tag}{'' if SEED == 0 else f'_s{SEED}'}.pt")
