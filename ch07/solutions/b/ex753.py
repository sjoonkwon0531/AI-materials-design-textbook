"""Ex 7.5.3: row space and spectrum of Delta W after 400 full-batch steps (NO early stopping), SGD(momentum) vs Adam."""
import numpy as np, torch
src = open("../../../ch07/labs/75_lora_bandgap.py").read().replace("../../ch07/", "../../../ch07/")  # lab paths are relative to ch07/labs
exec(src.split("methods = [")[0])
def energy(sv, ks=(1, 4, 8)):
    e = sv ** 2 / (sv ** 2).sum(); return [round(float(e[:k].sum()), 2) for k in ks]
for N in (50, 200):
    idx = np.random.default_rng(0).choice(tgt_pool, N, replace=False)
    perm = np.random.default_rng(10).permutation(idx); tr = perm[max(N // 5, 5):]
    Xt = Xs[tr].double(); S = torch.linalg.svdvals(Xt); Vh = torch.linalg.svd(Xt, full_matrices=False)[2]
    s_dim = int((S > S[0] * 1e-6).sum()); P = Vh[:s_dim].T @ Vh[:s_dim]
    for oname, lr in (("adam", 1e-3), ("sgd", 1e-2)):
        m = build("full", 0); params = list(m.parameters())
        opt = torch.optim.Adam(params, lr=lr) if oname == "adam" else torch.optim.SGD(params, lr=lr, momentum=0.9)
        for s in range(400):
            loss = (m(Xs[tr]).squeeze(-1) - ys[tr]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        dW1 = (m[0].weight - base_state["0.weight"]).detach().double()
        dW2 = (m[2].weight - base_state["2.weight"]).detach().double()
        sv1 = torch.linalg.svdvals(dW1).numpy(); sv2 = torch.linalg.svdvals(dW2).numpy()
        print(f"N={N} (train {len(tr)}, dim span(x) {s_dim}/132) {oname:4s} lr {lr:g}: train L1 {loss.item():.3f}, target-test {mae(m, tgt_te):.3f} | "
              f"layer1 |dW_outside|/|dW| {((dW1 - dW1 @ P).norm() / dW1.norm()).item():.1e}, rank(1e-6) {(sv1 > sv1[0]*1e-6).sum()}, "
              f"energy top1/4/8 {energy(sv1)} | layer2 rank(1e-6) {(sv2 > sv2[0]*1e-6).sum()}, energy top1/4/8 {energy(sv2)}", flush=True)
