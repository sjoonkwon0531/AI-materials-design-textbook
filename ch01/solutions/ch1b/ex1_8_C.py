"""Ex 1.8.C: forward vs reverse mode for f(x)=sum x_i^2 sin x_i, n=1000 (float64, CPU)."""
import time, torch
from torch.func import jvp, vjp, grad, vmap
torch.manual_seed(0); torch.set_num_threads(1)
n = 1000
x = torch.randn(n, dtype=torch.float64)
f = lambda x: (x**2*torch.sin(x)).sum()
exact = 2*x*torch.sin(x) + x**2*torch.cos(x)
def reverse():
    xr = x.clone().requires_grad_(True); f(xr).backward(); return xr.grad
def forward_loop():
    I = torch.eye(n, dtype=torch.float64); g = torch.empty(n, dtype=torch.float64)
    for i in range(n): g[i] = jvp(f, (x,), (I[i],))[1]
    return g
def forward_vmap():
    I = torch.eye(n, dtype=torch.float64)
    return vmap(lambda v: jvp(f, (x,), (v,))[1])(I)
def feval(): return f(x)
def bench(fn, rep):
    fn(); t = time.perf_counter()
    for _ in range(rep): out = fn()
    return (time.perf_counter()-t)/rep, out
tf, _ = bench(feval, 2000)
tr, gr = bench(reverse, 500)
tl, gl = bench(forward_loop, 3)
tv, gv = bench(forward_vmap, 20)
print(f"single f eval      : {tf*1e6:9.1f} us")
print(f"reverse (1 backward): {tr*1e6:9.1f} us  ({tr/tf:.1f} x f-eval), max err {(gr-exact).abs().max():.1e}")
print(f"forward, n JVP loop : {tl*1e6:9.1f} us  ({tl/tf:.0f} x f-eval), max err {(gl-exact).abs().max():.1e}")
print(f"forward, vmap(JVP)  : {tv*1e6:9.1f} us  ({tv/tf:.0f} x f-eval), max err {(gv-exact).abs().max():.1e}")
print(f"loop/reverse ratio = {tl/tr:.0f}, vmap/reverse ratio = {tv/tr:.1f}")
print(f"flop-ish work: forward-vmap processes an n x n = {n*n:,} tangent matrix vs reverse O(n)={n}")
