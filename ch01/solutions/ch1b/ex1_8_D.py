"""Ex 1.8.D: mini PINN for u_t = 0.1 u_xx on x in [0,1], t in [0,1],
u(x,0)=sin(pi x), u(0,t)=u(1,t)=0.  Exact u = exp(-0.1 pi^2 t) sin(pi x).
'4-layer MLP' = 4 Linear layers (2-32-32-32-1, tanh)."""
import time, math, torch
torch.manual_seed(0); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
D = 0.1
net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                          torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
u = lambda x, t: net(torch.cat([x, t], 1))
def residual(x, t):
    x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
    uu = u(x, t)
    u_t = torch.autograd.grad(uu.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(uu.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t - D*u_xx
Nf, Nb = 1000, 200
xf, tf = torch.rand(Nf, 1), torch.rand(Nf, 1)
x0 = torch.rand(Nb, 1); t0 = torch.zeros(Nb, 1)
tb = torch.rand(Nb, 1); xb = torch.cat([torch.zeros(Nb//2, 1), torch.ones(Nb//2, 1)])
def loss_fn():
    lr = residual(xf, tf).pow(2).mean()
    li = (u(x0, t0) - torch.sin(math.pi*x0)).pow(2).mean()
    lb = u(xb, tb).pow(2).mean()
    return lr + li + lb, (lr.item(), li.item(), lb.item())
xg, tg = torch.meshgrid(torch.linspace(0, 1, 101), torch.linspace(0, 1, 101), indexing='ij')
Xg, Tg = xg.reshape(-1, 1), tg.reshape(-1, 1)
U_ex = torch.exp(-D*math.pi**2*Tg)*torch.sin(math.pi*Xg)
def rel_l2():
    with torch.no_grad(): return ((u(Xg, Tg)-U_ex).norm()/U_ex.norm()).item()
t_start = time.time()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for it in range(1, 3001):
    opt.zero_grad(); L, parts = loss_fn(); L.backward(); opt.step()
    if it in (1, 1000, 2000, 3000): print(f"Adam it={it}: loss={L.item():.3e} (pde,ic,bc)={tuple(f'{p:.1e}' for p in parts)}, relL2={rel_l2():.2e}")
opt2 = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15,
                         history_size=50, line_search_fn="strong_wolfe")
def closure():
    opt2.zero_grad(); L, _ = loss_fn(); L.backward(); return L
opt2.step(closure)
L, parts = loss_fn()
print(f"after L-BFGS: loss={L.item():.3e} (pde,ic,bc)={tuple(f'{p:.1e}' for p in parts)}, relL2={rel_l2():.2e}")
with torch.no_grad():
    err = (u(Xg, Tg)-U_ex).abs(); print(f"max abs err on 101x101 grid = {err.max().item():.2e}")
    for tt in [0.0, 0.5, 1.0]:
        val = u(torch.tensor([[0.5]]), torch.tensor([[tt]])).item()
        print(f"  u(0.5,{tt}) PINN={val:.6f}, exact={math.exp(-D*math.pi**2*tt):.6f}")
print(f"wall time {time.time()-t_start:.0f} s")
