"""Lab 9.4-B (part 2): few-shot adaptation of the pre-trained FNO to an equation with a term it never saw,
    u_t + c u_x + beta (u^2/2)_x = nu u_xx + rho (u - u^3),   rho ~ U[1, 3].
Methods: zero-shot | scratch | last layer | LoRA (all nn.Linear layers) | full fine-tuning,
with the SAME number of optimiser steps for every method (compute-matched), K = 4, 16, 64, 224 trajectories.
The learning rate of each method is chosen for each K on 32 held-out pool trajectories (subset 0).
Run: python 94c_fewshot.py   (about 9 CPU-min)"""
import sys, time, importlib, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "../../ch09/labs")
C = importlib.import_module("94_common"); P = importlib.import_module("94b_pretrain")
STEPS, BS, LRS = 400, 32, (1e-3, 3e-3, 1e-2)


class LoRALinear(nn.Module):
    """y = x W^T + b + x A^T B^T ;  W, b frozen; B = 0 so Delta W = 0 at start (same class as Lab 7.17)."""
    def __init__(self, lin, r):
        super().__init__()
        self.lin = lin
        for p in self.lin.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, lin.in_features) / lin.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.enabled = True
    def forward(self, x):
        out = self.lin(x)
        return out + (x @ self.A.T) @ self.B.T if self.enabled else out


def pretrained():
    m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH); m.load_state_dict(torch.load(C.D + "94_fno_member0.pt")); return m


def add_lora(m, r):
    for p in m.parameters(): p.requires_grad_(False)        # spectral weights R stay frozen
    m.lift = LoRALinear(m.lift, r); m.proj1 = LoRALinear(m.proj1, r); m.proj2 = LoRALinear(m.proj2, r)
    for blk in m.blocks: blk.W = LoRALinear(blk.W, r)
    return m


def adapt(method, sub, seed, lr, steps=STEPS):
    torch.manual_seed(1000 + seed)
    if method == "scratch":
        m = C.TFNO(P.WIDTH, P.MODES, P.DEPTH)
    else:
        m = pretrained()
        if method == "last":
            for p in m.parameters(): p.requires_grad_(False)
            for p in m.proj2.parameters(): p.requires_grad_(True)
        elif method.startswith("lora"):
            add_lora(m, int(method[4:]))
    params = [p for p in m.parameters() if p.requires_grad]
    C.train(m, sub, steps, bs=BS, lr=lr, seed=seed, params=params)
    return m, sum((2 if p.is_complex() else 1) * p.numel() for p in params)


def subset(pool, K, s):
    idx = np.random.default_rng(10 * K + s).choice(224, K, replace=False)        # training subsets: pool[0:224]
    return dict(u=pool["u"][idx], coef=pool["coef"][idx])


if __name__ == "__main__":
    pool, te, te_id = [P.load(n) for n in ("react_train", "react_test", "pre_test")]
    val = dict(u=pool["u"][224:], coef=pool["coef"][224:])                       # validation: pool[224:256]
    base = pretrained()
    e0, eid0 = C.traj_error(base, te), C.traj_error(base, te_id)
    print(f"pretrained member 0: ID test {100 * eid0.mean():.2f}% | zero-shot on the reaction family {100 * e0.mean():.2f}%")
    triv = C.rel_l2(torch.from_numpy(np.repeat(te["u"][:, :1], 10, 1)), torch.from_numpy(te["u"][:, 4::4]))
    print(f"trivial baseline u(x, t) = u0(x): {100 * triv.mean():.1f}%")
    six = dict(u=te["u"][:6], coef=te["coef"][:6])
    ex = dict(true=te["u"][:6][:, [0, 20, 40]], zero=C.predict(base, six, [20, 40]).numpy(), coef=te["coef"][:6])
    err = lambda m, d: 100 * C.traj_error(m, d).mean().item()
    out, lr_tab = {}, {}; t_all = time.process_time()
    for K in (4, 16, 64, 224):
        seeds = (0, 1, 2) if K < 224 else (0,)
        print(f"--- K = {K} labelled trajectories ({K * 40} space-time fields), {STEPS} steps, {len(seeds)} subset(s) ---", flush=True)
        best = {}
        for meth in ("scratch", "lora4", "full"):            # learning rate: validation error on subset 0
            runs = {lr: adapt(meth, subset(pool, K, 0), 0, lr) for lr in LRS}
            ev = {lr: err(m, val) for lr, (m, _) in runs.items()}
            lr = min(ev, key=ev.get); best[meth] = (lr, runs[lr]); lr_tab[(K, meth)] = [ev[l] for l in LRS]
            print(f"  lr selection {meth:8s} " + "  ".join(f"{l:.0e}: {v:5.1f}%" for l, v in ev.items()) + f"   -> {lr:.0e}", flush=True)
        best["lora1"] = (best["lora4"][0], None); best["last"] = (1e-2, None)
        for meth in ("scratch", "last", "lora1", "lora4", "full"):
            lr, cached = best[meth]; errs, forget = [], []
            for s in seeds:
                t0 = time.process_time()
                m, ntr = cached if (s == 0 and cached is not None) else adapt(meth, subset(pool, K, s), s, lr)
                cpu = time.process_time() - t0
                errs.append(err(m, te)); forget.append(err(m, te_id))
                if K == 64 and s == 0:
                    ex[meth] = C.predict(m, six, [20, 40]).numpy()
                    if meth == "lora4":
                        for mod in m.modules():
                            if isinstance(mod, LoRALinear): mod.enabled = False
                        off = err(m, te_id)
            out[(K, meth)] = (np.mean(errs), np.std(errs), np.mean(forget), ntr, lr)
            fg = "      -" if meth == "scratch" else f"{np.mean(forget):6.2f} %"
            print(f"  {meth:8s} trainable {ntr:6d}  lr {lr:.0e}  target-test error {np.mean(errs):6.2f} +- {np.std(errs):4.2f} %"
                  f"   ID-test error after adaptation {fg}   ({cpu:.0f} CPU-s per run)", flush=True)
    print(f"LoRA r=4 (K=64, subset 0) with the adapters switched off: ID-test error {off:.2f} % (= pretrained)")
    print(f"total {time.process_time() - t_all:.0f} CPU-s")
    np.savez(C.D + "94c_fewshot.npz", keys=np.array([f"{k[0]}|{k[1]}" for k in out]), vals=np.array(list(out.values())),
             zero=100 * e0.mean().item(), triv=100 * triv.mean().item(), **{"ex_" + k: v for k, v in ex.items()})
