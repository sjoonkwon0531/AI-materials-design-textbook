"""Ex 5.7.5: chi from coexistence (eq. 5.7.4), mean-field chi_MF = pi rho Delta a / 15, and the pair-correlation
weighting int phi g d^3r / int phi d^3r of a one-component DPD fluid at rho = 3 and rho = 5."""
import sys, time, math, numpy as np
sys.path.insert(0, "../../../ch05/labs")
from dpd_basics import simulate
from dpd_core import make_amat
phi = 0.05; chi = math.log((1-phi)/phi)/(1-2*phi)
print(f"(a) phi=0.05 -> chi = ln(19)/0.9 = {chi:.4f}")
for rho, lit in ((3, 0.286), (5, 0.689)):
    mf = math.pi*rho/15
    print(f"(b) rho={rho}: chi_MF/Delta a = {mf:.4f}; literature {lit}; ratio MF/lit = {mf/lit:.3f}; lit/MF = {lit/mf:.3f}")
t0 = time.time(); edges = np.linspace(0, 1.0, 51); rc = 0.5*(edges[1:]+edges[:-1])
for rho, a in ((3, 25.0), (5, 15.0), (5, 25.0)):
    L = 6.0; n = int(rho*L**3); typ = np.zeros(n, np.int64)
    x, v, o = simulate((L, L, L), typ, make_amat(a, 0.0), 4.5, 3.0, 0.02, 0.65, 10.0, 0.0, 5, 11)
    from dpd_core import run, forces
    f = np.zeros_like(x); pt = np.zeros(3); NOB = np.zeros((0, 2), np.int64); box = np.array([L, L, L])
    forces(x, v, typ, make_amat(a, 0.0), 4.5, 3.0, 0.02, box, NOB, 0.0, f, pt)
    H = np.zeros(50); ns = 0
    for s in range(100):
        run(x, v, f, typ, make_amat(a, 0.0), 4.5, 3.0, 0.02, 0.65, box, NOB, 0.0, 50, 10**6, pt)
        d = x[:, None, :] - x[None, :, :]; d -= L*np.round(d/L); r = np.sqrt((d**2).sum(-1))[np.triu_indices(n, 1)]
        H += np.histogram(r, edges)[0]; ns += 1
    shell = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
    g = H/ns/(0.5*n*rho*shell)
    w = rc**2*(1-rc)**2
    R = np.sum(w*g)/np.sum(w)
    print(f"rho={rho}, a={a}: g(0.05)={g[2]:.3f}, g(0.25)={g[12]:.3f}, g(0.5)={g[25]:.3f}, g(0.85)={g[42]:.3f}; "
          f"<g>_phi = int phi g / int phi = {R:.3f} -> chi_MF*<g> = {math.pi*rho/15*R:.3f} Delta a", flush=True)
print(f"wall {time.time()-t0:.0f} s")
