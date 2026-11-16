"""Ex 1.6.E: fixed-point layer z = tanh(Wz + Ux + b): fixed-point iteration vs Newton,
dependence on spectral radius of W."""
import numpy as np
rng = np.random.default_rng(0)
n, m = 50, 10
W0 = rng.standard_normal((n, n))/np.sqrt(n)
W0 /= np.max(np.abs(np.linalg.eigvals(W0)))        # spectral radius 1
U = rng.standard_normal((n, m))/np.sqrt(m); b = 0.1*rng.standard_normal(n)
x = rng.standard_normal(m)
def run(rho, tol=1e-12, maxit=2000):
    W = rho*W0; c = U@x + b
    G = lambda z: z - np.tanh(W@z + c)
    z = np.zeros(n); kfp = None
    for k in range(1, maxit+1):
        z = np.tanh(W@z + c)
        if np.linalg.norm(G(z)) < tol: kfp = k; break
    z = np.zeros(n); kn = None; hist = []
    for k in range(1, 100):
        s = 1 - np.tanh(W@z + c)**2
        J = np.eye(n) - s[:, None]*W
        z = z - np.linalg.solve(J, G(z))
        r = np.linalg.norm(G(z)); hist.append(r)
        if r < tol: kn = k; break
    # local contraction factor at solution: spectral radius of diag(s) W
    s = 1 - np.tanh(W@z + c)**2
    rloc = np.max(np.abs(np.linalg.eigvals(s[:, None]*W)))
    return kfp, kn, rloc, np.linalg.norm(W, 2), hist
print(" rho  ||W||_2  rho(diag(s)W)  FP iters  Newton iters")
for rho in [0.3, 0.5, 0.8, 0.9, 0.95, 1.2, 1.5, 2.0, 3.0]:
    kfp, kn, rloc, nrm, hist = run(rho)
    print(f"{rho:4.2f}  {nrm:6.3f}   {rloc:8.4f}      {str(kfp):>6}   {str(kn):>6}")
_, _, _, _, hist = run(0.9)
print("Newton residuals rho=0.9:", ["%.1e" % h for h in hist])
# rho=3: Newton with backtracking line search on ||G||
W = 3.0*W0; c = U@x + b
G = lambda z: z - np.tanh(W@z + c)
z = np.zeros(n)
for k in range(1, 200):
    s = 1 - np.tanh(W@z + c)**2
    dz = np.linalg.solve(np.eye(n) - s[:, None]*W, -G(z)); lam = 1.0
    while np.linalg.norm(G(z+lam*dz)) > (1-1e-4*lam)*np.linalg.norm(G(z)) and lam > 1e-8: lam *= 0.5
    z = z + lam*dz
    if np.linalg.norm(G(z)) < 1e-12: break
print(f"rho=3 damped Newton: iters={k}, |G|={np.linalg.norm(G(z)):.1e}")
