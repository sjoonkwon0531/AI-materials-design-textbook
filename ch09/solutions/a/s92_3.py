"""Ex 9.2.3: Deep Ritz vs strong-form residual for -u'' = pi^2 sin(pi x), u = x(1-x) N(x)."""
import math, time, sys
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
PI = math.pi; J_EX = -PI**2 / 4
xe = torch.linspace(0, 1, 1001).unsqueeze(1); ue = torch.sin(PI * xe)
xq = ((torch.arange(20000) + 0.5) / 20000).unsqueeze(1)                    # fine midpoint rule for the "true" energy

def make(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
u_of = lambda net, x: x * (1 - x) * net(x)
f = lambda x: PI**2 * torch.sin(PI * x)

def energy(net, x):
    x = x.clone().requires_grad_(True); u = u_of(net, x)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return (0.5 * ux**2 - f(x) * u).mean()

def strong(net, x):
    x = x.clone().requires_grad_(True); u = u_of(net, x)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    return (-uxx - f(x)).pow(2).mean()

def run(kind, pts, seed, iters=2000, lr=1e-3):
    net = make(seed); opt = torch.optim.Adam(net.parameters(), lr=lr); g = torch.Generator().manual_seed(100 + seed)
    grid = ((torch.arange(256) + 0.5) / 256).unsqueeze(1); fixed_rand = torch.rand(256, 1, generator=g)
    lossf = energy if kind == "ritz" else strong
    c0 = time.process_time(); errs = {}
    for it in range(iters):
        x = grid if pts == "grid" else fixed_rand if pts == "fixed" else torch.rand(256, 1, generator=g)
        L = lossf(net, x); opt.zero_grad(); L.backward(); opt.step()
        if it + 1 in (2000, 5000, 20000):
            with torch.no_grad(): errs[it + 1] = float((u_of(net, xe) - ue).norm() / ue.norm())
    cpu = (time.process_time() - c0) / iters
    xl = grid if pts == "grid" else fixed_rand
    Jtrain = float(energy(net, xl)) if pts != "fresh" else float("nan"); Jtrue = float(energy(net, xq))
    with torch.no_grad(): err = float((u_of(net, xe) - ue).norm() / ue.norm()); emax = float((u_of(net, xe) - ue).abs().max())
    return err, emax, cpu * 1e3, Jtrain, Jtrue, errs

print(f"exact J = -pi^2/4 = {J_EX:.6f}; midpoint-256 energy of the exact solution: {float((0.5*(PI*torch.cos(PI*((torch.arange(256)+0.5)/256)))**2 - PI**2*torch.sin(PI*((torch.arange(256)+0.5)/256))**2).mean()):.6f}")
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
for kind, pts in (("strong", "grid"), ("ritz", "grid"), ("ritz", "fresh"), ("ritz", "fixed")):
    R = [run(kind, pts, s, iters) for s in range(3)]
    e = np.array([r[0] for r in R]); 
    print(f"{kind:6s} {pts:5s} ({iters} Adam): rel L2 = {e.mean():.2e} +/- {e.std():.1e} ({', '.join(f'{v:.2e}' for v in e)}), max err {np.mean([r[1] for r in R]):.2e}, {np.mean([r[2] for r in R]):.2f} ms/iter, "
          f"J on training points {', '.join(f'{r[3]:.5f}' for r in R)}; true J(u_theta) {', '.join(f'{r[4]:.6f}' for r in R)} (J - J_exact: {', '.join(f'{r[4]-J_EX:+.1e}' for r in R)})"
          + (f"; err history seed0 {R[0][5]}" if iters > 2000 else ""), flush=True)
