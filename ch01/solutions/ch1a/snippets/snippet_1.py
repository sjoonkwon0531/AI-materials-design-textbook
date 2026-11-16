import numpy as np, torch
f = lambda x: np.sin(5*x)*np.cos(x**2); xt = np.linspace(-2, 2, 801)
def net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1,64), torch.nn.Tanh(), torch.nn.Linear(64,64),
                               torch.nn.Tanh(), torch.nn.Linear(64,1)).double()
def train(n0, x, y, dt, steps=4000):
    m = net(0); m.load_state_dict(n0.state_dict()); m = m.to(dt)
    X, Y = (torch.tensor(v[:, None], dtype=dt) for v in (x, y))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad(); ((m(X)-Y)**2).mean().backward(); opt.step()
    with torch.no_grad(): return m(torch.tensor(xt[:, None], dtype=dt)).double().numpy().ravel()
P64, P32 = [], []
for r in range(8):
    g = np.random.default_rng(100+r); x = g.uniform(-2, 2, 128); y = f(x) + g.normal(0, .05, 128)
    n0 = net(1000+r); P64.append(train(n0, x, y, torch.float64)); P32.append(train(n0, x, y, torch.float32))
P64, P32 = np.array(P64), np.array(P32); var = P64.var(0, ddof=1)
bias2 = (P64.mean(0) - f(xt))**2 - var/8; prec = ((P32 - P64)**2).mean(0)
