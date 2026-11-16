"""Ex 9.3.5: FNO with alias-free activations (Fourier upsample x r -> GELU -> low-pass -> downsample)."""
import os, sys, time, math, importlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
LAB = "../ch09/labs"; sys.path.insert(0, LAB); M = importlib.import_module("93_models")
d = np.load(os.path.join(LAB, "..", "data", "93_burgers.npz"))
traj = torch.tensor(d["traj"]); NTR, NTE, NF = 1000, 200, traj.shape[-1]
A, U = traj[:, 0], traj[:, -1]; tr, te = slice(0, NTR), slice(2000, 2000 + NTE)
sub = lambda X, s: X[:, None, ::NF // s]
RES = [32, 64, 128, 256, 512]; EPOCHS, BATCH = 50, 20

class AAAct(nn.Module):
    def __init__(s, r): super().__init__(); s.r = r
    def forward(s, v):
        if s.r == 1: return F.gelu(v)
        N = v.shape[-1]
        return M.fourier_interp(F.gelu(M.fourier_interp(v, s.r * N)), N)

def make(r):
    net = M.FNO1d(width=32, kmax=16, depth=4)
    if r: a = AAAct(r); net.act = a; net.proj[1] = a; net.aa = a
    return net

def train(net, s=64, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed); a, u = sub(A[tr], s), sub(U[tr], s); c0 = time.process_time()
    M.fit(net, lambda idx: M.rel_l2(net(a[idx]), u[idx]), NTR, epochs, BATCH, lr=3e-3, seed=seed)
    return time.process_time() - c0

@torch.no_grad()
def test(net, s): return M.rel_l2(net(sub(A[te], s)), sub(U[te], s)).item()

@torch.no_grad()
def bl_check(net, label):
    abl = M.fourier_interp(M.fourier_interp(A[te][:, None], 16), NF); ref = net(abl)
    raw = [M.rel_l2(net(abl[..., ::NF // s]), ref[..., ::NF // s]).item() for s in (32, 64, 128)]
    lp = [M.rel_l2(net(abl[..., ::NF // s]), M.fourier_interp(ref, s)).item() for s in (32, 64, 128)]
    # low modes only (|k| < 8... compare the first 9 Fourier coefficients)
    lo = []
    for s in (32, 64, 128):
        ph = torch.fft.rfft(net(abl[..., ::NF // s]))[..., :9] / s; rh = torch.fft.rfft(ref)[..., :9] / NF
        lo.append(((ph - rh).abs().pow(2).sum(-1).sqrt() / rh.abs().pow(2).sum(-1).sqrt()).mean().item())
    print(f"  [{label}] band-limited input (|k|<8), grid s vs 512 (pointwise): " + "  ".join(f"s={s}: {x:.1e}" for s, x in zip((32, 64, 128), raw))
          + " | vs low-passed 512 output: " + "  ".join(f"{x:.1e}" for x in lp) + " | modes k<=8 only: " + "  ".join(f"{x:.1e}" for x in lo))

which = sys.argv[1]
if which == "base":
    sd = torch.load(os.path.join(LAB, "93_fno_burgers.pt")); net = make(0); net.load_state_dict(sd[64])
    print("baseline FNO (Lab 9.8 weights, trained at 64): test " + "  ".join(f"{s}: {test(net, s):.4f}" for s in RES)); bl_check(net, "plain GELU")
    # timing of 150 updates each
    for r in (0, 2, 4):
        torch.manual_seed(0); n = make(r); a, u = sub(A[tr], 64), sub(U[tr], 64); opt = torch.optim.Adam(n.parameters(), 1e-3)
        c0 = time.process_time()
        for it in range(150):
            idx = torch.randint(0, NTR, (BATCH,)); loss = M.rel_l2(n(a[idx]), u[idx]); opt.zero_grad(); loss.backward(); opt.step()
        cu = (time.process_time() - c0) / 150
        with torch.no_grad():
            c0 = time.process_time()
            for _ in range(20): n(sub(A[te], 64))
            ci = (time.process_time() - c0) / 20 / NTE
        print(f"  upsampling x{r if r else 1}: {cu*1e3:.1f} ms CPU per update (batch 20, s=64); inference {ci*1e3:.3f} ms per sample")
else:
    r = int(which); torch.manual_seed(0); net = make(r); cpu = train(net)
    print(f"alias-free FNO, upsampling x{r}, trained at s=64 ({cpu:.0f} s CPU): test " + "  ".join(f"{s}: {test(net, s):.4f}" for s in RES), flush=True)
    bl_check(net, f"AA x{r}")
    torch.save(net.state_dict(), f"../ch09/solutions/a/s93_5_aa{r}.pt")
    with torch.no_grad():
        # residual aliasing of the activation: same weights, same grid, different upsampling factors (reference x8)
        for s in (32, 64, 128):
            a = sub(A[te], s); net.aa.r = 8; ref = net(a); row = []
            for rr in (1, 2, 4):
                net.aa.r = rr; row.append(M.rel_l2(net(a), ref).item())
            net.aa.r = r
            print(f"  same weights on grid {s}: ||G_r - G_8|| / ||G_8||  r=1 (plain GELU): {row[0]:.1e}  r=2: {row[1]:.1e}  r=4: {row[2]:.1e}")
        for rr in (1, 4, 8):
            net.aa.r = rr; print(f"  weights trained with x{r}, evaluated with x{rr}: test (s=64) {test(net, 64):.4f}"); bl_check(net, f"AA x{r} weights, eval x{rr}")
        net.aa.r = r
        a64 = sub(A[te], 64); eq = M.rel_l2(net(torch.roll(a64, 5, -1)), torch.roll(net(a64), 5, -1)).item(); print(f"  translation equivariance error (shift 5/64): {eq:.1e}")
        # high-wavenumber content of the output at 512
        p = net(sub(A[te], 512))[:, 0]; eh = torch.fft.rfft(p - U[te]).abs().pow(2).mean(0).sqrt() / NF; th = torch.fft.rfft(U[te]).abs().pow(2).mean(0).sqrt() / NF
        print("  spectrum at 512: " + "; ".join(f"k={k}: truth {th[k]:.2e} err {eh[k]:.2e}" for k in (8, 24, 40, 64, 96)))
