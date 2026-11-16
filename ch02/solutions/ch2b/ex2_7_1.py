# Ex 2.7.1: Lab 2.15 PINN with 80 vs 8 collocation points; errors on a fine grid over several seeds
import numpy as np
from scipy.optimize import least_squares
H = 20
def fwd(p, x):
    w, b, v, c = p[:H], p[H:2*H], p[2*H:3*H], p[3*H]
    a = np.tanh(np.outer(x, w) + b)
    return a@v + c, (-2*a*(1-a**2))@(v*w**2)
xf = np.linspace(0, 1, 2001)
def train(nc, seed, method):
    xc = np.linspace(0, 1, nc); f = np.pi**2*np.sin(np.pi*xc)
    def residual(p):
        u, uxx = fwd(p, xc); u0, _ = fwd(p, [0.]); u1, _ = fwd(p, [1.])
        return np.concatenate([-uxx-f, np.sqrt(10)*np.array([u0[0], u1[0]])])
    p0 = 0.5*np.random.default_rng(seed).standard_normal(3*H+1)
    sol = least_squares(residual, p0, method=method, max_nfev=20000 if method == 'trf' else None)
    u, uxx = fwd(sol.x, xf)
    res_f = np.abs(-uxx - np.pi**2*np.sin(np.pi*xf)).max()       # PDE residual between collocation points
    return sol.cost, np.abs(u-np.sin(np.pi*xf)).max(), res_f

for nc, method in [(80, 'lm'), (8, 'trf')]:
    print(f"\ncollocation points = {nc} (residuals {nc+2} vs 61 params; method={method})")
    errs = []
    for seed in range(6):
        cost, err, rf = train(nc, seed, method); errs.append(err)
        print(f"  seed {seed}: train cost={cost:.1e}  max|u-u_exact| on 2001 pts={err:.2e}  max PDE residual on fine grid={rf:.2e}")
    print(f"  spread of max error over seeds: {min(errs):.1e} .. {max(errs):.1e}")
try:
    least_squares(lambda p: np.zeros(10), np.zeros(61), method='lm')
except Exception as ex:
    print("\nmethod='lm' with 10 residuals < 61 params ->", type(ex).__name__, ":", str(ex)[:90])
# FEM reference (P1, uniform, same problem as Lab 2.8): predictable O(h^2)
import io, contextlib, importlib
with contextlib.redirect_stdout(io.StringIO()):
    m = importlib.import_module('ex2_4_5')
for N in [8, 80]:
    x = m.mesh(N, 'uniform'); u = m.fem(x)
    print(f"FEM N={N:3d}: L2 err={m.l2(x, u):.3e}, L2/h^2={m.l2(x, u)*N*N:.3f}")
