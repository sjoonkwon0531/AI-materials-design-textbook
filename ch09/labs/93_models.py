"""Sec.9.3 models, written from scratch in PyTorch (CPU):  FNO-1D, DeepONet, a periodic CNN baseline.

Import (file name starts with a digit):
    import importlib; M = importlib.import_module("93_models")
    net = M.FNO1d(width=32, kmax=16, depth=4)          # (B, in_ch, N) -> (B, out_ch, N), any N
    net = M.DeepONet(m=64, p=64)                       # (B, m) sensors, (Q, 1) query points -> (B, Q)

Conventions
  * fields live on a uniform periodic grid x_j = j L / N, tensors are (batch, channel, N)
  * kmax = highest retained wavenumber index (modes 0..kmax of rfft are learned, all others are set to zero)
"""
import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
#  Fourier neural operator (1D)
# ----------------------------------------------------------------------------------------------
class SpectralConv1d(nn.Module):
    """(K v)(x) = F^-1[ R(k) . F v ](x),  R(k) in C^{cout x cin} for k = 0..kmax, zero above."""

    def __init__(self, cin, cout, kmax):
        super().__init__()
        self.cin, self.cout, self.kmax = cin, cout, kmax
        s = 1.0 / (cin * cout)
        self.weight = nn.Parameter(s * torch.randn(cin, cout, kmax + 1, dtype=torch.cfloat))

    def forward(self, v):                                    # v: (B, cin, N)
        N = v.shape[-1]
        vh = torch.fft.rfft(v)                               # (B, cin, N//2+1)
        m = min(self.kmax + 1, N // 2 + 1)
        out = torch.zeros(v.shape[0], self.cout, N // 2 + 1, dtype=torch.cfloat, device=v.device)
        out[..., :m] = torch.einsum("bik,iok->bok", vh[..., :m], self.weight[..., :m])
        return torch.fft.irfft(out, n=N)


class FNO1d(nn.Module):
    """u = Q o (sigma(W_L + K_L)) o ... o (sigma(W_1 + K_1)) o P (a).

    grid      : append the coordinate x/L as an input channel (breaks translation equivariance)
    residual  : output = a[:, :out_ch] + net(a)          (one-step maps)
    conserve  : None | "mean" (zero mode of the increment removed) | "flux" (increment = -d/dx of a learned flux)
    """

    def __init__(self, in_ch=1, out_ch=1, width=32, kmax=16, depth=4, grid=False,
                 residual=False, conserve=None, L=1.0, act=F.gelu):
        super().__init__()
        self.grid, self.residual, self.conserve, self.L, self.act, self.out_ch = grid, residual, conserve, L, act, out_ch
        self.lift = nn.Conv1d(in_ch + int(grid), width, 1)
        self.K = nn.ModuleList([SpectralConv1d(width, width, kmax) for _ in range(depth)])
        self.W = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, out_ch, 1))

    def features(self, a):
        if self.grid:
            x = torch.arange(a.shape[-1], device=a.device, dtype=a.dtype) / a.shape[-1]
            a = torch.cat([a, x.expand(a.shape[0], 1, -1)], 1)
        v = self.lift(a)
        for i, (K, W) in enumerate(zip(self.K, self.W)):
            v = K(v) + W(v)
            if i < len(self.K) - 1: v = self.act(v)
        return v

    def forward(self, a):                                    # a: (B, in_ch, N)
        out = self.proj(self.features(a))
        if self.conserve == "mean":
            out = out - out.mean(-1, keepdim=True)
        elif self.conserve == "flux":                        # out is a flux J;  increment = -dJ/dx (spectral)
            N = a.shape[-1]
            k = 2 * math.pi / self.L * torch.arange(N // 2 + 1, device=a.device)
            Jh = torch.fft.rfft(out)
            if N % 2 == 0: Jh = torch.cat([Jh[..., :-1], torch.zeros_like(Jh[..., -1:])], -1)
            out = torch.fft.irfft(-1j * k * Jh, n=N)
        return a[:, :self.out_ch] + out if self.residual else out


class CNN1d(nn.Module):
    """Periodic CNN baseline with the same lifting/projection as FNO1d; local kernels of `ksize` grid points."""

    def __init__(self, in_ch=1, out_ch=1, width=32, ksize=5, depth=4):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, width, 1)
        self.C = nn.ModuleList([nn.Conv1d(width, width, ksize, padding=ksize // 2, padding_mode="circular") for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, out_ch, 1))

    def forward(self, a):
        v = self.lift(a)
        for i, C in enumerate(self.C):
            v = C(v)
            if i < len(self.C) - 1: v = F.gelu(v)
        return self.proj(v)


# ----------------------------------------------------------------------------------------------
#  DeepONet (unstacked):  G(u)(y) ~ sum_k b_k(u(x_1..x_m)) t_k(y) + b0
# ----------------------------------------------------------------------------------------------
def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act: layers.append(act())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """branch: R^m -> R^p (sensor values);  trunk: y -> R^p (basis functions evaluated at the query points).

    periodic > 0 : trunk input is [cos 2 pi j y, sin 2 pi j y], j = 1..periodic  (exactly periodic basis)
    basis        : optional fixed (Q, p) matrix of basis functions on the output grid (POD-DeepONet);
                   then the trunk net is not used and the query grid is fixed.
    """

    def __init__(self, m, p=64, width=128, depth=3, periodic=4, basis=None, mean=None):
        super().__init__()
        self.periodic, self.p = periodic, p
        self.branch = mlp([m] + [width] * (depth - 1) + [p])
        d_in = 2 * periodic if periodic else 1
        self.trunk = None if basis is not None else mlp([d_in] + [width] * (depth - 1) + [p], last_act=True)
        self.register_buffer("basis", basis); self.register_buffer("mean", mean)
        self.b0 = nn.Parameter(torch.zeros(1))

    def trunk_features(self, y):                             # y: (Q, 1) in [0, 1)
        if self.basis is not None: return self.basis
        if self.periodic:
            j = torch.arange(1, self.periodic + 1, device=y.device, dtype=y.dtype)
            y = torch.cat([torch.cos(2 * math.pi * y * j), torch.sin(2 * math.pi * y * j)], -1)
        return self.trunk(y)                                 # (Q, p)

    def forward(self, u, y):                                 # u: (B, m), y: (Q, 1)
        out = self.branch(u) @ self.trunk_features(y).T + self.b0
        return out + self.mean if self.mean is not None else out


# ----------------------------------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------------------------------
def rel_l2(pred, true):
    """mean over the batch of ||pred - true||_2 / ||true||_2 (norm over all non-batch dims)."""
    d = (pred - true).flatten(1).norm(dim=1); n = true.flatten(1).norm(dim=1)
    return (d / n).mean()


def rel_l2_each(pred, true):
    return (pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)


def count_params(net):
    """number of real parameters (a complex weight counts as two)."""
    return sum(2 * p.numel() if p.is_complex() else p.numel() for p in net.parameters() if p.requires_grad)


def fit(net, step_loss, n_train, epochs, batch=32, lr=2e-3, wd=1e-5, seed=0, log=None):
    """Adam + one-cycle cosine schedule.  step_loss(idx) -> scalar loss for the sample indices idx."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(n_train / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time(); hist = []
    for ep in range(epochs):
        perm = torch.randperm(n_train, generator=g); tot = 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            loss = step_loss(idx)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        hist.append(tot / n_train)
        if log and (ep + 1) % log == 0: print(f"   epoch {ep+1:4d}  train loss {hist[-1]:.4e}  ({time.time()-t0:.0f} s)")
    return hist, time.time() - t0


def fourier_interp(u, N_out):
    """band-limited (trigonometric) interpolation of a periodic grid function (..., N) -> (..., N_out)."""
    N = u.shape[-1]
    uh = torch.fft.rfft(u)
    out = torch.zeros(*u.shape[:-1], N_out // 2 + 1, dtype=uh.dtype)
    m = min(N // 2 + 1, N_out // 2 + 1)
    out[..., :m] = uh[..., :m]
    if N_out > N and N % 2 == 0: out[..., N // 2] *= 0.5     # split the Nyquist mode
    return torch.fft.irfft(out, n=N_out) * (N_out / N)
