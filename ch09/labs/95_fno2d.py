"""FNO-2D written from scratch in PyTorch (used by Lab 9.5 (b) and the Case Study of 9.6).

Architecture (after Li et al., ICLR 2021, without batch normalisation):
    lifting  P : (a(x), x1, x2)  ->  R^width          (pointwise linear)
    L Fourier layers:  v <- sigma( F^{-1}( R . F v ) + W v )   (no sigma after the last one)
    projection Q : R^width -> 128 -> 1                 (pointwise MLP)
The network predicts the INCREMENT of a time step: u_next = u + Q(...).
With conserve=True the mean (the k = 0 Fourier mode) of the increment is removed, so the
spatial mean of the field is conserved exactly, by construction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """v -> F^{-1}( R . F v ): keep the lowest `modes` wavenumbers in each direction and
    multiply the channel vector of every retained mode by its own complex matrix R(k1, k2).
    Tensors are channel-last, (B, N1, N2, C)."""

    def __init__(self, width, modes):
        super().__init__()
        self.width, self.modes = width, modes
        scale = 1.0 / (width * width)
        # two blocks of retained modes: k1 in [0, modes) and k1 in [-modes, 0); k2 in [0, modes) (rfft).
        # weight layout (k1, k2, in, out): one batched matrix product per block, which on a CPU is
        # an order of magnitude faster than the textbook einsum("bixy,ioxy->boxy").
        self.w_pos = nn.Parameter(scale * torch.rand(modes, modes, width, width, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.rand(modes, modes, width, width, dtype=torch.cfloat))

    @staticmethod
    def mul(vh, w):                                          # vh: (B, m, m, C), w: (m, m, C, C)
        return torch.matmul(vh.permute(1, 2, 0, 3), w).permute(2, 0, 1, 3)

    def forward(self, v):                                    # v: (B, N1, N2, C), real
        B, N1, N2, C = v.shape
        m = self.modes
        vh = torch.fft.rfft2(v, dim=(1, 2))                  # (B, N1, N2//2 + 1, C)
        out = torch.zeros(B, N1, N2 // 2 + 1, C, dtype=vh.dtype, device=v.device)
        out[:, :m, :m] = self.mul(vh[:, :m, :m], self.w_pos)
        out[:, -m:, :m] = self.mul(vh[:, -m:, :m], self.w_neg)
        return torch.fft.irfft2(out, s=(N1, N2), dim=(1, 2))


class FNO2d(nn.Module):
    def __init__(self, modes=12, width=20, n_layers=4, in_channels=1, proj=64, conserve=False):
        super().__init__()
        self.modes, self.width, self.n_layers, self.conserve = modes, width, n_layers, conserve
        self.lift = nn.Linear(in_channels + 2, width)
        self.spec = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(n_layers)])
        self.lin = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.proj1 = nn.Linear(width, proj)
        self.proj2 = nn.Linear(proj, 1)

    @staticmethod
    def grid(B, N1, N2, device, dtype):
        x1 = torch.arange(N1, device=device, dtype=dtype) / N1
        x2 = torch.arange(N2, device=device, dtype=dtype) / N2
        g1 = x1.view(1, N1, 1, 1).expand(B, N1, N2, 1)
        g2 = x2.view(1, 1, N2, 1).expand(B, N1, N2, 1)
        return g1, g2

    def increment(self, u):                                  # u: (B, N1, N2), one field per sample
        B, N1, N2 = u.shape
        g1, g2 = self.grid(B, N1, N2, u.device, u.dtype)
        v = self.lift(torch.cat([u.unsqueeze(-1), g1, g2], dim=-1))    # lifting P
        for l in range(self.n_layers):
            v = self.spec[l](v) + self.lin[l](v)                       # F^{-1}(R F v) + W v
            if l < self.n_layers - 1:
                v = F.gelu(v)
        d = self.proj2(F.gelu(self.proj1(v))).squeeze(-1)              # projection Q
        if self.conserve:
            d = d - d.mean(dim=(-2, -1), keepdim=True)       # remove the k = 0 mode of the increment
        return d

    def forward(self, u):
        return u + self.increment(u)                         # next state = current state + increment


def to_double(model):
    """Cast a trained model to float64 / complex128 (nn.Module.double() leaves complex weights untouched)."""
    model.double()
    for mod in model.modules():
        if isinstance(mod, SpectralConv2d):
            mod.w_pos.data = mod.w_pos.data.to(torch.cdouble)
            mod.w_neg.data = mod.w_neg.data.to(torch.cdouble)
    return model


def rel_l2_loss(pred, target):
    """Mean over the batch of ||pred - target||_2 / ||target||_2."""
    num = (pred - target).flatten(1).norm(dim=1)
    den = target.flatten(1).norm(dim=1)
    return (num / den).mean()


def n_params(model):
    """Number of real parameters (a complex weight counts as two)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


if __name__ == "__main__":
    import time
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for width in (20, 24, 32):
        net = FNO2d(modes=12, width=width)
        x = torch.randn(20, 64, 64)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        t0 = time.perf_counter()
        for _ in range(5):
            opt.zero_grad(); loss = rel_l2_loss(net(x), x); loss.backward(); opt.step()
        tt = (time.perf_counter() - t0) / 5
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(5):
                net(x)
            ti = (time.perf_counter() - t0) / 5
            x1 = x[:1]
            t0 = time.perf_counter()
            for _ in range(50):
                net(x1)
            t1 = (time.perf_counter() - t0) / 50
        print(f"width {width}: {n_params(net):,d} real parameters; train step (batch 20) {1e3 * tt:.0f} ms; "
              f"inference {1e3 * ti / 20:.2f} ms/sample (batch 20), {1e3 * t1:.2f} ms (batch 1)")
