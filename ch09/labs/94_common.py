"""Shared code for the §9.4 labs (owner C).

PDE family on the periodic domain x in [0, 2*pi):

    u_t + c u_x + beta (u^2/2)_x = nu u_xx + rho (u - u^3)

  * solver  : pseudo-spectral ETDRK4 (Kassam-Trefethen contour-integral coefficients), 2/3 dealiasing
  * model   : time-conditioned FNO-1D with a hard initial-condition ansatz u = u0 + t * NN(u0, coeffs, t)
  * residual: label-free PDE residual of the prediction (spectral in x, forward-mode AD in t)

Every lab sets torch.set_num_threads(1) (2 shared CPU cores).
"""
import math, time
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(1)
L = 2 * math.pi
D = "../ch09/data/"


# --------------------------------------------------------------------------------------------------
# 1. data: random initial conditions and the reference solver
# --------------------------------------------------------------------------------------------------
def sample_ic(n_samp, n, rng, kmax=5, amp=1.0, decay=1.0):
    """Random periodic initial condition: mean + sum_{k<=kmax} k^-decay (a_k cos kx + b_k sin kx),
    rescaled so that max|u0 - mean| = amp * U(0.5, 1.0); mean ~ U(-0.3, 0.3)."""
    x = np.arange(n) * L / n
    k = np.arange(1, kmax + 1)
    a = rng.standard_normal((n_samp, kmax)) / k ** decay
    b = rng.standard_normal((n_samp, kmax)) / k ** decay
    u = a @ np.cos(np.outer(k, x)) + b @ np.sin(np.outer(k, x))
    u *= (amp * rng.uniform(0.5, 1.0, (n_samp, 1))) / np.abs(u).max(1, keepdims=True)
    return u + rng.uniform(-0.3, 0.3, (n_samp, 1))


def etdrk4(u0, nu, c, beta, rho, T=1.0, dt=5e-3, n_save=40, dealias=True):
    """ETDRK4 for u_t = Lin u + N(u), Lin = -nu k^2 - i c k, N = -beta (u^2/2)_x + rho (u - u^3).
    u0: (B, n) real; nu, c, beta, rho: (B,) arrays.  Returns (B, n_save+1, n) snapshots incl. t = 0."""
    B, n = u0.shape
    k = np.fft.rfftfreq(n, d=1.0 / n)                      # integer wavenumbers (domain length 2*pi)
    nu, c, beta, rho = [np.asarray(a, float).reshape(B, 1) for a in (nu, c, beta, rho)]
    Lin = -nu * k ** 2 - 1j * c * k                        # (B, nk)
    E, E2 = np.exp(dt * Lin), np.exp(dt * Lin / 2)
    M = 64                                                 # contour-integral evaluation of the phi functions
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)   # full circle: Lin is complex (advection)
    LR = dt * Lin[..., None] + r                           # (B, nk, M)
    Q = dt * ((np.exp(LR / 2) - 1) / LR).mean(-1)
    f1 = dt * ((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3).mean(-1)
    f2 = dt * ((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3).mean(-1)
    f3 = dt * ((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3).mean(-1)
    mask = (k <= n // 3) if dealias else np.ones_like(k, bool)

    def N(v):
        u = np.fft.irfft(v, n=n, axis=-1)
        return (-0.5j * k * beta * np.fft.rfft(u * u, axis=-1) + rho * np.fft.rfft(u - u ** 3, axis=-1)) * mask

    v = np.fft.rfft(u0, axis=-1)
    steps = int(round(T / dt)); every = steps // n_save
    assert every * n_save == steps
    out = np.empty((B, n_save + 1, n)); out[:, 0] = u0
    for s in range(1, steps + 1):
        Nv = N(v); a = E2 * v + Q * Nv
        Na = N(a); b = E2 * v + Q * Na
        Nb = N(b); cc = E2 * a + Q * (2 * Nb - Nv)
        Nc = N(cc)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if s % every == 0:
            out[:, s // every] = np.fft.irfft(v, n=n, axis=-1)
    return out


def sample_coeffs(n_samp, rng, kind="pre"):
    """Task priors.
    pre     : nu ~ logU[0.05, 0.2], c ~ U[-1, 1], beta ~ U[0, 1], rho = 0      (pre-training family)
    lownu   : nu ~ logU[0.02, 0.04]  (below the range), rest as 'pre'           (coefficient shift)
    react   : rho ~ U[1, 3], rest as 'pre'                                      (a term never seen)"""
    nu = np.exp(rng.uniform(np.log(0.05), np.log(0.2), n_samp))
    c = rng.uniform(-1, 1, n_samp); beta = rng.uniform(0, 1, n_samp); rho = np.zeros(n_samp)
    if kind == "lownu":
        nu = np.exp(rng.uniform(np.log(0.02), np.log(0.04), n_samp))
    elif kind == "react":
        rho = rng.uniform(1, 3, n_samp)
    elif kind != "pre":
        raise ValueError(kind)
    return nu, c, beta, rho


def make_dataset(n_samp, seed, kind="pre", n_solver=256, n_grid=128, amp=1.0, kmax=5, chunk=500):
    """Solve on n_solver points, keep every (n_solver//n_grid)-th point.  Returns dict of float32 arrays:
    u (N, 41, n_grid) with t_j = j/40, coef (N, 4) = (nu, c, beta, rho)."""
    rng = np.random.default_rng(seed)
    u0 = sample_ic(n_samp, n_solver, rng, kmax=kmax, amp=amp)
    nu, c, beta, rho = sample_coeffs(n_samp, rng, kind)
    us = [etdrk4(u0[i:i + chunk], nu[i:i + chunk], c[i:i + chunk], beta[i:i + chunk], rho[i:i + chunk])
          for i in range(0, n_samp, chunk)]
    u = np.concatenate(us)[:, :, ::n_solver // n_grid]
    return dict(u=u.astype(np.float32), coef=np.stack([nu, c, beta, rho], 1).astype(np.float32))


# --------------------------------------------------------------------------------------------------
# 2. model: time-conditioned FNO-1D (same Fourier layer as §9.3.2) with a hard initial condition
# --------------------------------------------------------------------------------------------------
class SpectralConv1d(nn.Module):
    """(K v)(x) = F^-1( R . F v )(x), keeping the lowest `modes` wavenumbers."""
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.R = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, v):                                   # v: (B, width, n)
        vh = torch.fft.rfft(v, dim=-1)
        out = torch.zeros_like(vh)
        out[..., :self.modes] = torch.einsum("bik,iok->bok", vh[..., :self.modes], self.R)
        return torch.fft.irfft(out, n=v.shape[-1], dim=-1)


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.K = SpectralConv1d(width, modes); self.W = nn.Linear(width, width)

    def forward(self, v):                                   # sigma( K v + W v ), residual connection
        return v + torch.nn.functional.gelu(self.K(v) + self.W(v.transpose(1, 2)).transpose(1, 2))


COEF_SCALE = torch.tensor([0.1, 1.0, 1.0, 1.0])            # nu enters as log(nu / 0.1)


class TFNO(nn.Module):
    """u_theta(x, t; u0, coef) = u0(x) + t * NN(u0, coef, t)(x):  the initial condition holds exactly."""
    def __init__(self, width=32, modes=16, depth=4):
        super().__init__()
        self.lift = nn.Linear(1 + 4 + 1, width)             # channels: u0, (log nu, c, beta, rho), t
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(depth)])
        self.proj1 = nn.Linear(width, 64); self.proj2 = nn.Linear(64, 1)
        with torch.no_grad():                               # a coefficient that is absent in pre-training enters
            self.lift.weight[:, 4].zero_()                  # through a zero column: no garbage at zero-shot

    def features(self, u0, coef, t):                        # u0 (B, n); coef (B, 4); t (B,)
        B, n = u0.shape
        cf = torch.cat([torch.log(coef[:, :1] / 0.1), coef[:, 1:]], 1)
        z = torch.cat([u0[..., None], cf[:, None, :].expand(B, n, 4), t[:, None, None].expand(B, n, 1)], -1)
        v = self.lift(z).transpose(1, 2)
        for blk in self.blocks:
            v = blk(v)
        return torch.nn.functional.gelu(self.proj1(v.transpose(1, 2)))

    def forward(self, u0, coef, t):
        return u0 + t[:, None] * self.proj2(self.features(u0, coef, t)).squeeze(-1)


def n_params(m, trainable=False):
    """Number of real parameters (a complex weight counts as two)."""
    return sum((2 if p.is_complex() else 1) * p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


def rel_l2(pred, true):
    """Relative L2 error per sample over all remaining axes."""
    d = (pred - true).reshape(len(pred), -1); t = true.reshape(len(true), -1)
    return d.norm(dim=1) / t.norm(dim=1)


def train(model, data, steps, bs=64, lr=3e-3, seed=0, log=None, params=None, wd=0.0, t_idx=None):
    """Adam + cosine schedule on random (trajectory, time) pairs; relative-L2 loss.  Returns wall seconds."""
    g = torch.Generator().manual_seed(seed)
    u = torch.from_numpy(data["u"]); coef = torch.from_numpy(data["coef"])
    N, nt, n = u.shape
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time(); model.train()
    for s in range(steps):
        i = torch.randint(0, N, (bs,), generator=g)
        j = torch.randint(1, nt, (bs,), generator=g)        # t_j = j / (nt - 1), j >= 1
        pred = model(u[i, 0], coef[i], j.float() / (nt - 1))
        loss = rel_l2(pred, u[i, j]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (s + 1) % log == 0:
            print(f"    step {s + 1:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    return time.time() - t0


@torch.no_grad()
def predict(model, data, t_idx, bs=256):
    """Predictions at snapshot indices t_idx (list).  Returns (N, len(t_idx), n) tensor."""
    u0 = torch.from_numpy(data["u"][:, 0]); coef = torch.from_numpy(data["coef"]); nt = data["u"].shape[1]
    out = []
    for j in t_idx:
        tj = torch.full((len(u0),), j / (nt - 1))
        out.append(torch.cat([model(u0[i:i + bs], coef[i:i + bs], tj[i:i + bs]) for i in range(0, len(u0), bs)]))
    return torch.stack(out, 1)


def traj_error(model, data, t_idx=tuple(range(4, 41, 4))):
    """Per-trajectory relative L2 error over the space-time block t = 0.1, 0.2, ..., 1.0."""
    pred = predict(model, data, list(t_idx))
    return rel_l2(pred, torch.from_numpy(data["u"][:, list(t_idx)]))


# --------------------------------------------------------------------------------------------------
# 3. label-free PDE residual of a prediction
# --------------------------------------------------------------------------------------------------
def rhs_torch(u, coef, dealias=True):
    """nu u_xx - c u_x - beta (u^2/2)_x + rho (u - u^3), spectral derivatives on the grid of u (B, n)."""
    n = u.shape[-1]
    k = torch.fft.rfftfreq(n, d=1.0 / n)
    nu, c, beta, rho = [coef[:, i:i + 1] for i in range(4)]
    uh = torch.fft.rfft(u, dim=-1)
    u2h = torch.fft.rfft(u * u, dim=-1)
    if dealias:
        u2h = u2h * (k <= n // 3)
    lin = torch.fft.irfft((-nu * k ** 2 - 1j * c * k) * uh, n=n, dim=-1)
    return lin - beta * torch.fft.irfft(0.5j * k * u2h, n=n, dim=-1) + rho * (u - u ** 3)


def pde_residual(model, u0, coef, t):
    """r = d/dt u_theta - RHS(u_theta).  The time derivative is exact (forward-mode AD through the network)."""
    f = lambda tt: model(u0, coef, tt)
    with torch.no_grad():
        u, ut = torch.func.jvp(f, (t,), (torch.ones_like(t),))
        return u, ut - rhs_torch(u, coef)


def residual_norms(model, data, t_idx=tuple(range(4, 41, 4)), bs=256):
    """Per-trajectory RMS residual over the block t = 0.1 ... 1.0 (no labels used)."""
    u0 = torch.from_numpy(data["u"][:, 0]); coef = torch.from_numpy(data["coef"]); nt = data["u"].shape[1]
    r2 = torch.zeros(len(u0))
    for j in t_idx:
        tj = torch.full((len(u0),), j / (nt - 1))
        for i in range(0, len(u0), bs):
            _, r = pde_residual(model, u0[i:i + bs], coef[i:i + bs], tj[i:i + bs])
            r2[i:i + bs] += (r ** 2).mean(1)
    return (r2 / len(t_idx)).sqrt()
