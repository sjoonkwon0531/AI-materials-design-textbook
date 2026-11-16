"""Ex 9.2.3(c) follow-up: quadrature error of the fixed random points and a longer run."""
import math, importlib.util, sys
import numpy as np, torch
sys.argv = ["x"]; 
src = open("../ch09/solutions/a/s92_3.py").read().split("print(f\"exact J")[0]
exec(src)
PI = math.pi
def Jh_exact(x):
    return float((0.5 * (PI * torch.cos(PI * x))**2 - PI**2 * torch.sin(PI * x)**2).mean())
for seed in range(3):
    g = torch.Generator().manual_seed(100 + seed); xr = torch.rand(256, 1, generator=g)
    print(f"seed {seed}: J_h(u_exact) on the fixed random points = {Jh_exact(xr):.4f} (exact {J_EX:.4f})")
vals = [Jh_exact(torch.rand(256, 1, generator=torch.Generator().manual_seed(1000 + i))) for i in range(2000)]
print(f"std of J_h(u_exact) over 2000 random 256-point sets: {np.std(vals):.3f}, mean {np.mean(vals):.4f}")
# longer run with fixed random points, seed 0
seed = 0; net = make(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3); g = torch.Generator().manual_seed(100 + seed); xr = torch.rand(256, 1, generator=g)
for it in range(10000):
    L = energy(net, xr); opt.zero_grad(); L.backward(); opt.step()
    if it + 1 in (500, 2000, 5000, 10000):
        with torch.no_grad(): err = float((u_of(net, xe) - ue).norm() / ue.norm())
        print(f"fixed random points, seed 0, iter {it+1}: J_h = {float(energy(net, xr).detach()):.4f}, true J = {float(energy(net, xq).detach()):.4f}, rel L2 {err:.3f}")
