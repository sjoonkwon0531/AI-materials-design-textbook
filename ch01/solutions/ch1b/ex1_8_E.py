"""Ex 1.8.E: LJ force analytic vs autograd (reduced units eps=sigma=1)."""
import torch, numpy as np
torch.set_default_dtype(torch.float64)
eps_, sig = 1.0, 1.0
def E_pair(r1, r2):
    r = torch.linalg.norm(r1 - r2); sr6 = (sig/r)**6
    return 4*eps_*(sr6**2 - sr6)
def F_analytic(r1, r2):        # force on atom 1
    rv = r1 - r2; r = np.linalg.norm(rv); sr6 = (sig/r)**6
    return 24*eps_/r**2*(2*sr6**2 - sr6)*rv
maxerr = 0.0; rows = []
for r in np.linspace(0.9, 3.0, 22):
    d = np.array([1.0, 2.0, -0.5]); d /= np.linalg.norm(d)
    p1 = np.array([0.3, -0.2, 0.1]) + r*d; p2 = np.array([0.3, -0.2, 0.1])
    R = torch.tensor(np.stack([p1, p2]), requires_grad=True)
    E = E_pair(R[0], R[1]); F = -torch.autograd.grad(E, R)[0].numpy()
    Fa = F_analytic(p1, p2)
    rel = np.abs(F[0]-Fa).max()/max(np.abs(Fa).max(), 1e-300)
    maxerr = max(maxerr, np.abs(F[0]-Fa).max())
    rows.append((r, E.item(), np.dot(Fa, d), rel, np.abs(F[0]+F[1]).max()))
for r, E, Fr, rel, mom in rows[::3]:
    print(f"r={r:.2f}: E={E:+.6f}, F_r(analytic)={Fr:+.6e}, rel diff AD vs analytic={rel:.1e}, |F1+F2|={mom:.1e}")
print(f"max |F_AD - F_analytic| over 22 r values = {maxerr:.2e}")
rmin = 2**(1/6); R = torch.tensor([[rmin, 0, 0], [0, 0, 0.]], requires_grad=True)
E = E_pair(R[0], R[1]); F = -torch.autograd.grad(E, R)[0]
print(f"r_min=2^(1/6)={rmin:.6f}: E={E.item():.12f}, |F|={F.abs().max().item():.1e}")
# 2nd derivative via autograd: k = E"(r_min) = 36*2^(2/3) eps/sigma^2
r = torch.tensor(rmin, requires_grad=True)
Er = lambda r: 4*((1/r)**12 - (1/r)**6)
g = torch.autograd.grad(Er(r), r, create_graph=True)[0]; k = torch.autograd.grad(g, r)[0]
print(f"k = E''(r_min) by AD = {k.item():.10f}, analytic 36*2^(2/3)= {36*2**(2/3):.10f}")
