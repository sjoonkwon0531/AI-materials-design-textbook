"""Ex 7.5.3 (SGD vs Adam: rank / row space of Delta W) and 7.5.5 (L2-SP sweep vs LoRA), Lab 7.17 setup."""
import numpy as np, torch, json, time
src = open("../../../ch07/labs/75_lora_bandgap.py").read().replace("../../ch07/", "../../../ch07/")  # lab paths are relative to ch07/labs
exec(src.split("methods = [")[0])
def finetune2(m, idx, seed, steps=400, lr=1e-3, opt_name="adam", lam=0.0, theta0=None):
    params = [p for p in m.parameters() if p.requires_grad]
    perm = np.random.default_rng(10 + seed).permutation(idx)
    nv = max(len(idx) // 5, 5); va, tr = perm[:nv], perm[nv:]
    opt = torch.optim.Adam(params, lr=lr) if opt_name == "adam" else torch.optim.SGD(params, lr=lr, momentum=0.9)
    best = (mae(m, va), {k: v.clone() for k, v in m.state_dict().items()})
    for s in range(steps):
        loss = (m(Xs[tr]).squeeze(-1) - ys[tr]).abs().mean()
        if lam > 0:
            loss = loss + lam * sum(((p - p0) ** 2).sum() for p, p0 in zip(m.parameters(), theta0))
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 10 == 0:
            v = mae(m, va)
            if v < best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()})
    m.load_state_dict(best[1])
    return m, tr
# ---------------- 7.5.3
print("=== Ex 7.5.3 ===")
def energy(sv, ks=(1, 4, 8)):
    e = sv ** 2 / (sv ** 2).sum(); return [round(float(e[:k].sum()), 3) for k in ks]
for N in (50, 200):
    idx = np.random.default_rng(0).choice(tgt_pool, N, replace=False)
    for oname, lr in (("adam", 1e-3), ("sgd", 1e-2), ("sgd", 3e-2)):
        m = build("full", 0)
        m, tr = finetune2(m, idx, 0, lr=lr, opt_name=oname)
        dW1 = (m[0].weight - base_state["0.weight"]).detach().double()
        dW2 = (m[2].weight - base_state["2.weight"]).detach().double()
        Xt = Xs[tr].double()                                   # layer-1 inputs (fixed during training)
        U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
        s_dim = int((S > S[0] * 1e-10).sum())
        P = Vh[:s_dim].T @ Vh[:s_dim]                          # projector on the row space of the inputs
        out = (dW1 - dW1 @ P).norm() / dW1.norm()
        sv1 = torch.linalg.svdvals(dW1).numpy(); sv2 = torch.linalg.svdvals(dW2).numpy()
        r1 = int((sv1 > sv1[0] * 1e-6).sum())
        print(f"N={N} {oname} lr={lr}: n_train={len(tr)}, dim span(x)={s_dim}; target-test MAE {mae(m, tgt_te):.3f}; "
              f"layer1: |dW1 outside span(x)|/|dW1| = {out:.2e}, numerical rank(1e-6) {r1}; "
              f"layer1 energy top1/4/8 {energy(sv1)}; layer2 energy top1/4/8 {energy(sv2)}; |dW2|_F {np.sqrt((sv2**2).sum()):.3f}", flush=True)
# ---------------- 7.5.5
print("=== Ex 7.5.5 ===")
theta_base = None
out = {}
for meth, lam in [("full", l) for l in (0.0, 1e-4, 1e-3, 1e-2, 1e-1)] + [(f"LoRA r={r}", 0.0) for r in (1, 2, 4, 8)]:
    tm, sm = [], []
    for seed in range(5):
        idx = np.random.default_rng(seed).choice(tgt_pool, 200, replace=False)
        m = build(meth, seed)
        theta0 = [p.detach().clone() for p in m.parameters()]
        m, _ = finetune2(m, idx, seed, lam=lam, theta0=theta0)
        tm.append(mae(m, tgt_te)); sm.append(mae(m, src_te))
    out[f"{meth}|{lam}"] = (tm, sm)
    print(f"{meth:9s} lambda={lam:7.0e}: target-test {np.mean(tm):.3f} +- {np.std(tm):.3f} | source-test {np.mean(sm):.3f} +- {np.std(sm):.3f}", flush=True)
json.dump(out, open("ex755.json", "w"))
print(f"runtime {time.time()-t0:.0f} s")
