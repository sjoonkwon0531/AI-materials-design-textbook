import numpy as np
# 8.3.1
a, b, rho = 100.0, 1.0, 0.1
L = lambda t: 0.5 * (a * t[0]**2 + b * t[1]**2); grad = lambda t: np.array([a * t[0], b * t[1]])
ang = np.linspace(0, 2*np.pi, 200001); worst = max(L(rho*np.array([np.cos(x), np.sin(x)])) for x in ang[::100])
print("8.3.1(a) worst", worst, "eq 8.13", 0.5*rho**2*a)
th = np.array([0.05, 0.5]); g = grad(th); e = rho * g / np.linalg.norm(g); gs = grad(th + e)
print("g", g, "|g|", np.linalg.norm(g), "eps", e, "g_SAM", gs, "L(th)", L(th), "L(th+eps)", L(th+e))
print("angles deg: g", np.degrees(np.arctan2(g[1], g[0])), "gSAM", np.degrees(np.arctan2(gs[1], gs[0])), "between", np.degrees(np.arccos(g@gs/np.linalg.norm(g)/np.linalg.norm(gs))))
print("ratio component g1/g2:", g[0]/g[1], gs[0]/gs[1], "amplification", gs/g)
# exact worst in ball around th (radius 0.1): maximize via dense angle search
best = max(((L(th + rho*np.array([np.cos(x), np.sin(x)])), x) for x in ang))
print("exact worst around th", best[0], "dir", np.cos(best[1]), np.sin(best[1]), " 1st-order eps gives", L(th+e))
# 8.3.4 numerical check: noise injection == ridge with lambda = n sigma^2 (sum form)
rng = np.random.default_rng(0); n, p, s = 50, 5, 0.3
X = rng.normal(size=(n, p)); w0 = rng.normal(size=p); y = X @ w0 + 0.1 * rng.normal(size=n)
lam = n * s**2; w_r = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
K = 20000; Xa = np.repeat(X, K, 0) + s * rng.normal(size=(n*K, p)); ya = np.repeat(y, K)
w_n = np.linalg.lstsq(Xa, ya, rcond=None)[0]
print("8.3.4 ridge", np.round(w_r, 4), "noise-aug LS", np.round(w_n, 4), "max diff", np.abs(w_r-w_n).max(), "OLS", np.round(np.linalg.lstsq(X, y, rcond=None)[0], 4))
# 8.3.5 rotameter
R = np.array([10, 30, 50, 70, 90.]); V = np.array([20.0, 52.1, 84.6, 118.3, 151.0]); n = 5
A = np.column_stack([R, np.ones(n)]); H = 2*A.T@A/n; ev = np.linalg.eigvalsh(H)
print("8.3.5 H", H, "eig", ev, "kappa", ev[1]/ev[0])
z = (R - R.mean())/R.std(); As = np.column_stack([z, np.ones(n)]); Hs = 2*As.T@As/n; evs = np.linalg.eigvalsh(Hs)
print(" std: sigma_R", R.std(), "H", Hs, "eig", evs)
z1 = (R - R.mean())/R.std(ddof=1); A1 = np.column_stack([z1, np.ones(n)]); print(" ddof=1 eig", np.linalg.eigvalsh(2*A1.T@A1/n))
x = np.linalg.lstsq(A, V, rcond=None)[0]; xs = np.linalg.lstsq(As, V, rcond=None)[0]
print(" fits", x, xs, "back-transform", xs[0]/R.std(), xs[1]-xs[0]*R.mean()/R.std())
Rt = np.array([0, 25, 60, 100.]); print(" predictions equal:", A[:0].shape, np.allclose(np.column_stack([Rt, np.ones(4)])@x, np.column_stack([(Rt-R.mean())/R.std(), np.ones(4)])@xs))
J = np.mean((A@x - V)**2); print(" min J", J, "LOO-CV both", )
# LOO CV identical
def loo(Am):
    e = []
    for i in range(n):
        m = np.arange(n) != i; c = np.linalg.lstsq(Am[m], V[m], rcond=None)[0]; e.append(Am[i]@c - V[i])
    return np.sqrt(np.mean(np.square(e)))
print(" LOO RMSE raw", loo(A), "standardized", loo(As))
# sharpness in a 'rho-ball' worst case: 1/2 rho^2 lam_max
for r in (0.1,):
    print(" worst loss increase rho=0.1: raw", 0.5*r**2*ev[1], "std", 0.5*r**2*evs[1])
# 8.2.1
g = 0.95; T = 8
print("8.2.1 final weight", g**(T-1), "intermediate sum", (1-g)*sum(g**(t-1) for t in range(1, T)), [round((1-g)*g**(t-1),4) for t in range(1,T)])
