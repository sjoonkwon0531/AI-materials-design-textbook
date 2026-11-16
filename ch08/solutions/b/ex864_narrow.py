"""Ex 8.6.4: DRO-bsf trained on narrow landscapes (K=6, c=0.1, 1,200 iterations) vs the Lab's wide-trained model."""
import os, time, numpy as np, torch
from common import lab, LABS
torch.set_num_threads(1)
M = lab("86_dro"); t0 = time.time()
torch.manual_seed(0); rng = np.random.default_rng(0)
net_n = M.DRO(); opt = torch.optim.Adam(net_n.parameters(), 1e-3)
for it in range(1, 1201):
    Gt = M.sample_gmm(128, rng, K=6, c=0.1)
    Y = M.run_dro(net_n, Gt, M.x0_batch(128, rng), M.f_torch)
    loss = -torch.cummax(Y, 1).values[:, 1:].sum(1).mean()
    opt.zero_grad(); loss.backward()
    for p in net_n.parameters(): p.grad.clamp_(-0.1, 0.1)
    opt.step()
    if it % 300 == 0: print(f"  narrow training iter {it}: mean best-of-50 {float(Y.max(1).values.mean()):.3f} ({time.time()-t0:.0f} s)", flush=True)
torch.save(net_n.state_dict(), "ex864_dro_narrow.pt")
net_w = M.DRO(); net_w.load_state_dict(torch.load(os.path.join(LABS, "86_dro.pt"))["bsf"])
def roll(net, G, x0, Fs, blind):
    gen = torch.Generator().manual_seed(0); x = torch.as_tensor(x0, dtype=torch.float32); state = None; ys = []
    with torch.no_grad():
        for t in range(51):
            y = M.f_torch(G, x); ys.append(y)
            if t == 50: break
            mean, L, state = net.step(x, torch.full_like(y, 0.5) if blind else y, state)
            x = (mean + (L @ torch.randn(x.shape, generator=gen)[..., None])[..., 0]).clamp(0, 1)
    Y = torch.stack(ys, 1).numpy(); return Fs - Y.max(1)
for lab_, c in (("wide (c=0.3)", 0.3), ("narrow (c=0.1)", 0.1)):
    r = np.random.default_rng(99); G = M.sample_gmm(150, r, K=6, c=c); Fs = M.global_max(G); x0 = M.x0_batch(150, r)
    U = np.random.default_rng(5).uniform(0, 1, (150, 50, 3))
    rs = Fs - np.maximum(M.f_np(G, x0), M.f_np(G, U).max(1))
    print(f"[test {lab_}] random search regret@50 {rs.mean():.3f} +/- {rs.std(ddof=1)/np.sqrt(150):.3f}", flush=True)
    for nm, net in (("trained wide", net_w), ("trained narrow", net_n)):
        a = roll(net, G, x0, Fs, False); b = roll(net, G, x0, Fs, True)
        print(f"   DRO-bsf {nm:<15s}: regret@50 {a.mean():.3f} +/- {a.std(ddof=1)/np.sqrt(150):.3f} | blindfolded {b.mean():.3f} | "
              f"blind - normal {b.mean()-a.mean():+.3f} (paired s.e. {(b-a).std(ddof=1)/np.sqrt(150):.3f})", flush=True)
print(f"wall {time.time()-t0:.0f} s")
