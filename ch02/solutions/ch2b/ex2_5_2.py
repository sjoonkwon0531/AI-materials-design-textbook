# Ex 2.5.2: Crank-Nicolson amplification factor, unconditional stability, g -> -1 oscillation
import numpy as np, sympy as sp
lam, th = sp.symbols('lambda theta', positive=True); g = sp.Symbol('g')
s = sp.sin(th/2)**2
# CN: u^{n+1} - lam/2 d2 u^{n+1} = u^n + lam/2 d2 u^n,  d2 e^{ij th} = -4 sin^2(th/2) e^{ij th}
eq = sp.Eq(g*(1 + lam/2*4*s), 1 - lam/2*4*s)
gcn = sp.solve(eq, g)[0]; print("g_CN =", sp.simplify(gcn))
print("1 - g_CN^2 =", sp.factor(sp.simplify(1 - gcn**2)), " (>=0 for lam>0)")
print("lim lam->oo at theta=pi:", sp.limit(gcn.subs(th, sp.pi), lam, sp.oo))
L, T = np.meshgrid(np.logspace(-3, 4, 400), np.linspace(0, np.pi, 400))
G = (1-2*L*np.sin(T/2)**2)/(1+2*L*np.sin(T/2)**2)
print(f"numeric max|g_CN| over lam in [1e-3,1e4], theta in [0,pi]: {np.abs(G).max():.6f}")
for l in [0.5, 1, 10, 100]:
    gc = (1-2*l)/(1+2*l); gb = 1/(1+4*l)
    print(f"lam={l:6g}: theta=pi  g_CN={gc:+.4f}   g_BTCS={gb:+.4f}")

# demo: heat eq, N=50 (h=0.02), IC u=1 inside, u=0 on boundary (incompatible), lam=10, 10 steps
N = 50; h = 1/N; lam_ = 10.0; n = N-1
T2 = (np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1))
I = np.eye(n)
D_cn = np.linalg.solve(I - lam_/2*T2, I + lam_/2*T2); D_be = np.linalg.inv(I - lam_*T2)
u_cn = np.ones(n); u_be = np.ones(n)
for k in range(1, 11):
    u_cn = D_cn@u_cn; u_be = D_be@u_be
    if k in (1, 2, 3, 10):
        sc = np.sum(np.diff(np.sign(np.diff(u_cn[:10]))) != 0)
        print(f"step {k:2d}: CN min={u_cn.min():+.4f} max={u_cn.max():.4f} (wiggles near wall: {sc})   BTCS min={u_be.min():+.4f} max={u_be.max():.4f}")
print(f"rho(D_CN)={max(abs(np.linalg.eigvals(D_cn))):.4f}, most negative eig={min(np.linalg.eigvals(D_cn).real):+.4f};  rho(D_BTCS)={max(abs(np.linalg.eigvals(D_be))):.4f}")
