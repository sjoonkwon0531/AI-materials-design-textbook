"""Ex 5.8.4: (a) Leibler spinodal for f=0.3 (continuous Debye functions);
(b) spinodal of a discrete Gaussian chain N=10, <b^2>=0.895 (and variants) for f=1/2."""
import numpy as np
from scipy.optimize import minimize_scalar
g = lambda f, x: 2*(f*x + np.exp(-f*x) - 1)/x**2
def NGam(x, f):
    saa = g(f, x); sbb = g(1-f, x); sab = 0.5*(g(1, x) - saa - sbb)
    return (saa + sbb + 2*sab)/(saa*sbb - sab**2)
for f in (0.5, 0.3, 0.2):
    r = minimize_scalar(lambda x: NGam(x, f), bounds=(0.3, 20), method="bounded", options=dict(xatol=1e-10))
    print(f"(a) continuous Gaussian f={f}: (q*Rg)^2={r.x:.4f}, q*Rg={np.sqrt(r.x):.4f}, (chi N)_s={NGam(r.x, f)/2:.4f}")
def disc(q, N, NA, b2):
    i = np.arange(N); P = np.exp(-q*q*b2*np.abs(i[:, None]-i[None, :])/6)
    A = i < NA; B = ~A
    saa = P[np.ix_(A, A)].sum()/N; sbb = P[np.ix_(B, B)].sum()/N; sab = P[np.ix_(A, B)].sum()/N
    return saa, sbb, sab
def spin(N, NA, b2):
    G = lambda q: (lambda s: (s[0]+s[1]+2*s[2])/(s[0]*s[1]-s[2]**2))(disc(q, N, NA, b2))
    r = minimize_scalar(G, bounds=(0.2, 4.0), method="bounded", options=dict(xatol=1e-10))
    i = np.arange(N); Rg2 = b2*np.mean(np.abs(i[:, None]-i[None, :]))*0  # placeholder replaced below
    Rg2 = b2*(N*N-1)/(6*N)                     # discrete Gaussian chain: Rg^2 = b^2 (N^2-1)/(6N)
    return r.x, N*G(r.x)/2, np.sqrt(Rg2)
for N, NA, b2, lab in ((10, 5, 0.895, "bond <b^2>=0.895 (exercise)"), (10, 5, 8.84/9, "b^2=<Re^2>/9=0.982"),
                       (10, 5, 0.0, None), (20, 10, 0.895, "N=20"), (40, 20, 0.895, "N=40"), (160, 80, 0.895, "N=160")):
    if lab is None: continue
    q, cn, Rg = spin(N, NA, b2)
    print(f"(b) discrete Gaussian N={N}, {lab}: q*={q:.4f}, Rg={Rg:.4f}, q*Rg={q*Rg:.4f}, chi_s N={cn:.4f}")
