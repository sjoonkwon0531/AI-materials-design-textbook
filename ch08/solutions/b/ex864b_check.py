import os, numpy as np, torch
from common import lab, LABS
torch.set_num_threads(1)
M = lab("86_dro")
net_n = M.DRO(); net_n.load_state_dict(torch.load("ex864_dro_narrow.pt"))
net_w = M.DRO(); net_w.load_state_dict(torch.load(os.path.join(LABS, "86_dro.pt"))["bsf"])
r = np.random.default_rng(99); G = M.sample_gmm(150, r, K=6, c=0.1); x0 = M.x0_batch(150, r)
for nm, net in (("wide", net_w), ("narrow", net_n)):
    gen = torch.Generator().manual_seed(0); x = torch.as_tensor(x0, dtype=torch.float32); state = None; xs = []
    with torch.no_grad():
        for t in range(50):
            y = M.f_torch(G, x); mean, L, state = net.step(x, y, state)
            raw = mean + (L @ torch.randn(x.shape, generator=gen)[..., None])[..., 0]
            x = raw.clamp(0, 1); xs.append(raw.numpy())
    X = np.array(xs)
    print(nm, "fraction of raw query coords outside [0,1]: %.3f" % np.mean((X < 0) | (X > 1)),
          "| mean |L| diag %.3f" % float(torch.diagonal(L, dim1=-2, dim2=-1).abs().mean()),
          "| corner fraction (all 3 coords clamped) %.3f" % np.mean(((X < 0) | (X > 1)).all(-1)))
for nm, net in (("wide", net_w), ("narrow", net_n)):
    out = []
    for blind in (False, True):
        gen = torch.Generator().manual_seed(0); x = torch.as_tensor(x0, dtype=torch.float32); state = None; xs = []
        with torch.no_grad():
            for t in range(50):
                y = M.f_torch(G, x); mean, L, state = net.step(x, torch.full_like(y, 0.5) if blind else y, state)
                x = (mean + (L @ torch.randn(x.shape, generator=gen)[..., None])[..., 0]).clamp(0, 1); xs.append(x.numpy())
        out.append(np.array(xs))
    Yv = M.f_np(G, out[0].transpose(1, 0, 2))
    print(nm, "max |x_normal - x_blind| = %.2e, mean = %.2e | sd of observed y within a run (narrow test): median %.3f" % (
        np.abs(out[0] - out[1]).max(), np.abs(out[0] - out[1]).mean(), np.median(Yv.std(1))))
