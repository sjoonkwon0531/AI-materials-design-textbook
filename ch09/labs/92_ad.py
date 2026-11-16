"""Lab 9.2-ad: (A) u_xx of a network by nested reverse-mode autodiff vs finite differences of the network;
(B) cost of the k-th input derivative (k = 1..4) by nested reverse mode and by nested forward mode (torch.func.jvp).
"""
import time, math
import numpy as np, torch, mpmath as mp

torch.set_num_threads(1)

# ---------------------------------------------------------------- A. accuracy
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 16), torch.nn.Tanh(),
                          torch.nn.Linear(16, 1)).double()
xs = torch.linspace(0, 1, 200, dtype=torch.float64)[:, None]
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
for it in range(1500):                                   # make the network non-trivial: fit sin(2 pi x)
    opt.zero_grad(); ((net(xs) - torch.sin(2 * math.pi * xs)) ** 2).mean().backward(); opt.step()
print(f"A. network 1-16-16-1 tanh fitted to sin(2 pi x): fit RMSE {float(((net(xs)-torch.sin(2*math.pi*xs))**2).mean().sqrt().detach()):.1e}")

W = [p.detach().numpy() for p in net.parameters()]
mp.mp.dps = 40
def u_mp(x):                                             # the same network in 40-digit arithmetic
    h = [mp.tanh(mp.mpf(float(W[0][i, 0])) * x + mp.mpf(float(W[1][i]))) for i in range(16)]
    h = [mp.tanh(sum(mp.mpf(float(W[2][i, j])) * h[j] for j in range(16)) + mp.mpf(float(W[3][i]))) for i in range(16)]
    return sum(mp.mpf(float(W[4][0, j])) * h[j] for j in range(16)) + mp.mpf(float(W[5][0]))
xp = np.linspace(0.05, 0.95, 19)
ref = np.array([float(mp.diff(u_mp, mp.mpf(float(x)), 2)) for x in xp])
print(f"   reference u_xx by 40-digit arithmetic at 19 points, max|u_xx| = {np.abs(ref).max():.2f}")

def uxx_ad(model, dtype):
    x = torch.tensor(xp[:, None], dtype=dtype, requires_grad=True)
    u = model(x)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return torch.autograd.grad(ux.sum(), x)[0].detach().double().numpy().ravel()

def uxx_fd(model, dtype, h):
    x = torch.tensor(xp[:, None], dtype=dtype)
    with torch.no_grad():
        return ((model(x + h) - 2 * model(x) + model(x - h)) / h**2).double().numpy().ravel()

import copy
net32 = copy.deepcopy(net).float()
relerr = lambda a: np.abs(a - ref).max() / np.abs(ref).max()
print(f"   autodiff  float64: max rel err {relerr(uxx_ad(net, torch.float64)):.1e}")
print(f"   autodiff  float32: max rel err {relerr(uxx_ad(net32, torch.float32)):.1e}")
print("   central FD of the network, max rel err:")
print("        h      float64    float32")
OUT = dict(ad64=relerr(uxx_ad(net, torch.float64)), ad32=relerr(uxx_ad(net32, torch.float32)), fd=[], cost=[])
for h in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]:
    e64, e32 = relerr(uxx_fd(net, torch.float64, h)), relerr(uxx_fd(net32, torch.float32, h)); OUT["fd"].append((h, e64, e32))
    print(f"     {h:7.0e}  {e64:9.1e}  {e32:9.1e}")

# ---------------------------------------------------------------- B. cost vs derivative order
torch.manual_seed(1)
big = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                          torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)).double()
N = 2000
X = torch.rand(N, 1, dtype=torch.float64); Tt = torch.rand(N, 1, dtype=torch.float64)
params = list(big.parameters())

def kth_reverse(k, train):
    x = X.clone().requires_grad_(True)
    d = big(torch.cat([x, Tt], 1))
    for _ in range(k):
        d = torch.autograd.grad(d.sum(), x, create_graph=True)[0]
    if train:
        torch.autograd.grad(d.pow(2).mean(), params, allow_unused=True)
    return d

def kth_forward(k, train):
    f = lambda x: big(torch.cat([x, Tt], 1))
    for _ in range(k):
        f = (lambda g: (lambda x: torch.func.jvp(g, (x,), (torch.ones_like(x),))[1]))(f)
    d = f(X)
    if train:
        torch.autograd.grad(d.pow(2).mean(), params, allow_unused=True)
    return d

def bench(fn, *a, rep=25):
    ts = []
    for _ in range(rep):
        t0 = time.process_time(); fn(*a); ts.append(time.process_time() - t0)
    return np.min(ts) * 1e3          # minimum: robust to other jobs on the shared cores

print(f"B. network 2-64-64-64-1 tanh (float64), N = {N} points, CPU time, minimum of 25 runs, single thread")
with torch.no_grad():
    t_fwd = bench(lambda: big(torch.cat([X, Tt], 1)))
print(f"   forward pass only: {t_fwd:.2f} ms")
print("   k   reverse-nested   (+param grad)   forward-nested(jvp)   (+param grad)   max|rev-fwd|")
for k in range(1, 5):
    a = bench(kth_reverse, k, False); b = bench(kth_reverse, k, True)
    c = bench(kth_forward, k, False); d = bench(kth_forward, k, True)
    diff = float((kth_reverse(k, False) - kth_forward(k, False)).detach().abs().max())
    OUT["cost"].append((k, a, b, c, d))
    print(f"   {k}   {a:8.2f} ms ({a/t_fwd:5.1f}x)  {b:8.2f} ms     {c:8.2f} ms ({c/t_fwd:5.1f}x)   {d:8.2f} ms     {diff:.1e}")
OUT["t_fwd"] = t_fwd
import json; json.dump(OUT, open("../../ch09/data/92_ad.json", "w"))
