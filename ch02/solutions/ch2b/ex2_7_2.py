# Ex 2.7.2: space-time PINN for u_t = u_xx, u(x,0)=sin(pi x), u(0,t)=u(1,t)=0, t in [0, 0.2]
#   u_theta(x,t) = sum_j v_j tanh(w_j x + s_j t + b_j) + c ; u_t, u_xx and the Jacobian are analytic
import numpy as np, time
from scipy.optimize import least_squares
H = 20; Tf = 0.2
def unpack(p): return p[:H], p[H:2*H], p[2*H:3*H], p[3*H:4*H], p[4*H]
def feats(p, x, t):
    w, s, b, v, c = unpack(p); a = np.tanh(np.outer(x, w) + np.outer(t, s) + b); d = 1 - a**2
    return w, s, b, v, c, a, d
def fwd(p, x, t):
    w, s, b, v, c, a, d = feats(p, x, t)
    return a@v + c, d@(v*s), (-2*a*d)@(v*w**2)                        # u, u_t, u_xx
def jac_u(p, x, t):
    w, s, b, v, c, a, d = feats(p, x, t); X, T = x[:, None], t[:, None]
    return np.hstack([v*d*X, v*d*T, v*d, a, np.ones((len(x), 1))])
def jac_pde(p, x, t):                                                 # d(u_t - u_xx)/dp
    w, s, b, v, c, a, d = feats(p, x, t); X, T = x[:, None], t[:, None]
    dd = -2*a*d; q = -2*a*d; dq = -2*d*(1 - 3*a**2)
    Jut = np.hstack([v*s*dd*X, v*d + v*s*dd*T, v*s*dd, s*d, np.zeros((len(x), 1))])
    Juxx = np.hstack([v*(2*w*q + w**2*dq*X), v*w**2*dq*T, v*w**2*dq, w**2*q, np.zeros((len(x), 1))])
    return Jut - Juxx
Xg, Tg = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, Tf, 21)); xc, tc = Xg.ravel(), Tg.ravel()
xi = np.linspace(0, 1, 41); tb = np.linspace(0, Tf, 21); wic = np.sqrt(10)
def residual(p):
    u, ut, uxx = fwd(p, xc, tc)
    return np.concatenate([ut - uxx, wic*(fwd(p, xi, 0*xi)[0] - np.sin(np.pi*xi)),
                           wic*fwd(p, 0*tb, tb)[0], wic*fwd(p, 0*tb+1, tb)[0]])
def jac(p):
    return np.vstack([jac_pde(p, xc, tc), wic*jac_u(p, xi, 0*xi), wic*jac_u(p, 0*tb, tb), wic*jac_u(p, 0*tb+1, tb)])
p_test = 0.5*np.random.default_rng(7).standard_normal(4*H+1); e = np.zeros_like(p_test)
Jfd = np.zeros((len(residual(p_test)), len(p_test)))
for i in range(len(p_test)):
    e[:] = 0; e[i] = 1e-6; Jfd[:, i] = (residual(p_test+e) - residual(p_test-e))/2e-6
print(f"analytic Jacobian check: max |J - J_fd| = {np.abs(jac(p_test)-Jfd).max():.1e}")
exact = lambda x, t: np.exp(-np.pi**2*t)*np.sin(np.pi*x)
xf, tf = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, Tf, 101)); x1 = np.linspace(0, 1, 101)
for seed in [0, 1, 2]:
    t0 = time.time(); p0 = 0.5*np.random.default_rng(seed).standard_normal(4*H+1)
    sol = least_squares(residual, p0, jac=jac, method='lm', max_nfev=5000)
    u = fwd(sol.x, xf.ravel(), tf.ravel())[0]; err = np.abs(u - exact(xf.ravel(), tf.ravel())).max()
    eT = np.abs(fwd(sol.x, x1, np.full(101, Tf))[0] - exact(x1, Tf)).max()
    print(f"seed {seed}: params={4*H+1}, residuals={len(residual(p0))}, cost={sol.cost:.2e}, nfev={sol.nfev}, "
          f"max err (space-time)={err:.2e}, max err at t=0.2={eT:.2e}, {time.time()-t0:.1f}s")
# reference: method of lines + Crank-Nicolson, N=20 cells, dt=0.01 (20 steps)
N = 20; h = 1/N; dt = 0.01; n = N-1; x = np.linspace(0, 1, N+1)[1:-1]
T2 = (np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1))/h**2; I = np.eye(n)
D = np.linalg.solve(I - dt/2*T2, I + dt/2*T2); u = np.sin(np.pi*x)
for _ in range(int(round(Tf/dt))): u = D@u
print(f"MOL+CN reference (N=20, dt=0.01, 20 steps): max nodal err at t=0.2 = {np.abs(u-exact(x, Tf)).max():.2e}")
