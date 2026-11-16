"""Ex 9.3.1-9.3.4: small calculations."""
import sys, math, importlib
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "../../../ch09/labs"); M = importlib.import_module("93_models")
PI = math.pi
print("== 9.3.1 ==")
for km in (16, 8):
    lift = 32 + 32; four = 2 * 32 * 32 * (km + 1); W = 32 * 32 + 32; proj = (32 * 64 + 64) + (64 + 1)
    print(f"kmax={km}: lift {lift}, per layer {four} + {W} = {four+W}, x4 = {4*(four+W)}, proj {proj}, total {lift+4*(four+W)+proj}; count_params {M.count_params(M.FNO1d(width=32, kmax=km, depth=4))}")
dv, km = 32, 12; n2 = 2 * dv * dv * (2 * km + 1) * (km + 1)
print(f"2D: modes {(2*km+1)*(km+1)}, spectral reals {n2}, with W {n2 + dv*dv + dv}; 1D same kmax: {2*dv*dv*(km+1) + dv*dv + dv}, ratio {(n2 + dv*dv + dv)/(2*dv*dv*(km+1) + dv*dv + dv):.1f}")
torch.manual_seed(0); net = M.FNO1d(width=32, kmax=16, depth=4); a = torch.randn(4, 1, 64)
y0 = net(a)
loss = y0.pow(2).mean(); loss.backward()
gi = [K.weight.grad[..., 0].imag.abs().max().item() for K in net.K]; gr = [K.weight.grad[..., 0].real.abs().max().item() for K in net.K]
print("max |grad| of Im R(0) per layer:", gi, " of Re R(0):", [f"{v:.2e}" for v in gr])
with torch.no_grad():
    for K in net.K: K.weight[..., 0] += 1j * torch.randn(32, 32)
    print(f"change of output after perturbing Im R(0) by N(0,1): {(net(a)-y0).abs().max().item():.1e}; dead parameters {4*32*32}")
    a32 = torch.randn(4, 1, 32); y = net(a32)
    for K in net.K: K.weight[..., 16] += 1j * torch.randn(32, 32)
    print(f"on a 32-point grid, perturbing Im R(16) (Nyquist) changes the output by {(net(a32)-y).abs().max().item():.1e}")
print("== 9.3.2 ==")
nu, T = 0.01, 1.0
for k in range(1, 8): print(f"k={k}: exp(-nu (2 pi k)^2 T) = {math.exp(-nu*(2*PI*k)**2*T):.3e}")
print(f"threshold k > {math.sqrt(math.log(1e6)/(nu*4*PI*PI*T)):.3f}")
k = np.arange(0, 257); lam = 625.0 / ((2 * PI * k) ** 2 + 25) ** 2; var = 4 * lam; var[0] = lam[0]
for K in (5, 16, 32):
    adv = math.sqrt(var[K + 1:].sum() / var.sum()); mult = np.exp(-nu * (2 * PI * k) ** 2 * T); vh = var * mult**2
    print(f"GRF input of Lab 9.8: keep k<={K} (p={2*K+1}): relative L2 truncation error  advection {adv:.3e}   heat {math.sqrt(vh[K+1:].sum()/vh.sum()):.3e}")
print("== 9.3.3 ==")
tot = PI**2 / 8
def tail(p): 
    j = np.arange(1, p + 1); return tot - (1.0 / (2 * j - 1.0) ** 2).sum()
for p in (8, 32, 128):
    r = tail(p) / tot; print(f"p={p}: rel squared error {r:.5f} (approx 2/(pi^2 p) = {2/(PI**2*p):.5f}), rel L2 {math.sqrt(r):.4f}")
p = 1
while tail(p) / tot > 1e-4: p += 1
print(f"1 % L2: p = {p} odd wavenumbers ({2*p} real basis functions), estimate 2/(pi^2 1e-4) = {2/(PI**2*1e-4):.0f}; highest wavenumber {2*p-1}")
N = 8192; x = np.arange(N) / N
for delta in (0.03, 0.01, 0.003):
    h = np.tanh(np.sin(2 * PI * x) / (2 * PI * delta)); c = np.abs(np.fft.rfft(h)) ** 2; c[1:] *= 2; c /= c.sum()
    odd = c[1::2]                                   # wavenumbers 1, 3, 5, ...
    res = [math.sqrt(odd[p:].sum()) for p in (8, 32, 128)]
    cum = np.sqrt(np.maximum(1 - np.cumsum(odd), 0)); p1 = int(np.argmax(cum < 0.01)) + 1
    print(f"tanh profile width delta={delta}: rel L2 error p=8: {res[0]:.4f}  p=32: {res[1]:.2e}  p=128: {res[2]:.2e};  p for 1 %: {p1} (p*delta = {p1*delta:.2f}); even-mode content {c[2::2].sum():.1e}")
print("== 9.3.4 ==")
nu, dt = 0.01, 0.1
g = lambda k: math.exp(-nu * (2 * PI * k) ** 2 * dt)
print(f"g(1) = {g(1):.5f}, g(16) = {g(16):.3e}, g(17) = {g(17):.3e}; g(1)^100 = {g(1)**100:.4e}, g(2)^100 = {g(2)**100:.3e}")
print(f"(a) (1.01)^100 - 1 = {1.01**100-1:.4f}; 1 - 0.99^100 = {1-0.99**100:.4f}; e - 1 = {math.e-1:.4f}; absolute k=1: g^100 * 1.705 = {g(1)**100*(1.01**100-1):.4f}; 0.9^100 = {0.9**100:.3e}; 0.9^10 = {0.9**10:.3f}")
print(f"(b) n = ln(1e7)/ln(1.02) = {math.log(1e7)/math.log(1.02):.1f}; with 1.0069: {math.log(1e7)/math.log(1.0069):.0f}")
# (d) g_inf of a freshly initialised residual FNO at u = 0
vals = []
for seed in range(20):
    torch.manual_seed(seed); net = M.FNO1d(width=32, kmax=24, depth=4, residual=True, L=64.0)
    J = torch.autograd.functional.jacobian(lambda u: net(u[None, None])[0, 0], torch.zeros(128))
    g0 = torch.fft.fft(J[:, 0]); hi = g0[25:65]
    vals.append(hi.real.mean().item()); flat = (hi.real.max() - hi.real.min()).item(); im = hi.imag.abs().max().item()
vals = np.array(vals); print(f"(d) g_inf at initialisation, 20 seeds: mean {vals.mean():.4f}, std {vals.std():.4f}, min {vals.min():.4f}, max {vals.max():.4f}; fraction > 1: {(vals>1).mean():.2f}; flatness {flat:.1e}, imag {im:.1e}")
