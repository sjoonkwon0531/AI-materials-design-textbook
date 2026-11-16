"""Lab 7.6(c): DistributedDataParallel on CPU (gloo backend, 2 processes) vs single-process training
with the same GLOBAL batch. Shows (1) that DDP all-reduces (averages) gradients, (2) that the final
parameters match single-process training to float32 round-off, (3) the throughput on this 2-core machine.
Run: python 76_ddp.py   (about 1-2 min)."""
import os, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD, B_GLOBAL, EPOCHS, LR, SEED = 2, 256, 4, 1e-3, 0


def load():
    df = pd.read_parquet(os.path.join(HERE, "..", "data", "67_features_clean.parquet"))
    feat = [c for c in df.columns if c not in ("formula", "target", "chemsys", "reduced")]
    X = np.clip(StandardScaler().fit_transform(df[feat].values), -6, 6).astype(np.float32)
    n = (len(X) // B_GLOBAL) * B_GLOBAL                        # drop the ragged tail: equal batches everywhere
    return torch.from_numpy(X[:n]), torch.from_numpy(df.target.values[:n].astype(np.float32))


def make_model(H):
    torch.manual_seed(SEED)                                    # identical initial weights on every rank
    return nn.Sequential(nn.Linear(149, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))


def loss_fn(m, xb, yb):
    return (m(xb).squeeze(-1) - yb).abs().mean()


def make_opt(params, name):
    return torch.optim.AdamW(params, lr=LR) if name == "adamw" else torch.optim.SGD(params, lr=0.05, momentum=0.9)


def single_process(X, y, H, oname):
    torch.set_num_threads(1)
    m = make_model(H); opt = make_opt(m.parameters(), oname)
    t0 = time.time()
    for ep in range(EPOCHS):
        g = torch.Generator(); g.manual_seed(SEED + ep)          # the same permutation DistributedSampler draws
        perm = torch.randperm(len(y), generator=g)
        for s in range(0, len(y), B_GLOBAL):
            idx = perm[s:s + B_GLOBAL]
            opt.zero_grad(); loss_fn(m, X[idx], y[idx]).backward(); opt.step()
    return m, time.time() - t0


def worker(rank, X, y, H, oname, port, ret):
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    model = DDP(make_model(H))                                 # broadcasts rank-0 weights, registers all-reduce hooks
    opt = make_opt(model.parameters(), oname)
    sampler = DistributedSampler(TensorDataset(X, y), num_replicas=WORLD, rank=rank, shuffle=True, seed=SEED,
                                 drop_last=True)
    dl = DataLoader(TensorDataset(X, y), batch_size=B_GLOBAL // WORLD, sampler=sampler)
    # --- (1) look inside one step: local gradient vs the all-reduced one ---
    xb, yb = next(iter(dl))
    with model.no_sync():                                     # backward WITHOUT communication
        model.zero_grad(); loss_fn(model, xb, yb).backward()
    g_local = model.module[0].weight.grad.clone()
    model.zero_grad(); loss_fn(model, xb, yb).backward()      # normal DDP backward: grads are averaged
    g_ddp = model.module[0].weight.grad.clone()
    g_sum = g_local.clone(); dist.all_reduce(g_sum)           # explicit sum of local grads over ranks
    if rank == 0:
        ret["local_norm0"] = g_local.norm().item()
        ret["ddp_vs_mean"] = (g_ddp - g_sum / WORLD).abs().max().item()
    else:
        ret["local_norm1"] = g_local.norm().item()
    model.zero_grad()
    # --- (2) training ---
    dist.barrier(); t0 = time.time()
    for ep in range(EPOCHS):
        sampler.set_epoch(ep)
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(model, xb, yb).backward(); opt.step()
    dist.barrier()
    if rank == 0:
        ret["ddp_time"] = time.time() - t0
        ret["params"] = [p.detach().clone() for p in model.module.parameters()]
        ret["n_per_rank"] = len(sampler)
    else:
        ret["params1"] = [p.detach().clone() for p in model.module.parameters()]
    dist.destroy_process_group()


if __name__ == "__main__":
    X, y = load()
    print("training rows %d, global batch %d = %d ranks x %d, %d epochs" % (len(y), B_GLOBAL, WORLD, B_GLOBAL // WORLD, EPOCHS))
    res = {}
    for H, oname, port in ((128, "adamw", 29517), (128, "sgd", 29519), (1024, "adamw", 29518)):
        m_single, t_single = single_process(X, y, H, oname)
        ret = mp.Manager().dict()
        mp.spawn(worker, args=(X, y, H, oname, port, ret), nprocs=WORLD, join=True)
        p_single = list(m_single.parameters())
        npar = sum(p.numel() for p in p_single)
        print("--- MLP 149-%d-%d-1 (%d parameters), %s ---" % (H, H, npar, "AdamW lr 1e-3" if oname == "adamw" else "SGD lr 0.05 momentum 0.9"))
        print("rank-local gradient norms (layer 1, first batch): rank0 %.4f, rank1 %.4f" % (ret["local_norm0"], ret["local_norm1"]))
        print("DDP gradient - mean of local gradients: max |diff| = %.1e" % ret["ddp_vs_mean"])
        d_ranks = max((a - b).abs().max().item() for a, b in zip(ret["params"], ret["params1"]))
        d_single = max((a - b.detach()).abs().max().item() for a, b in zip(ret["params"], p_single))
        scale = max(p.abs().max().item() for p in p_single)
        print("after training: max |theta_rank0 - theta_rank1| = %.1e" % d_ranks)
        print("after training: max |theta_DDP - theta_single(global batch)| = %.1e  (max |theta| = %.2f)" % (d_single, scale))
        n_seen = EPOCHS * len(y)
        print("wall: single process %.2f s (%.0f samples/s) | DDP 2 procs %.2f s (%.0f samples/s) | speed-up %.2f" % (
            t_single, n_seen / t_single, ret["ddp_time"], n_seen / ret["ddp_time"], t_single / ret["ddp_time"]))
        res[f"{H}_{oname}"] = dict(npar=npar, t_single=t_single, t_ddp=ret["ddp_time"], d_single=d_single, d_ranks=d_ranks)
    json.dump(res, open(os.path.join(HERE, "76_ddp.json"), "w"))
