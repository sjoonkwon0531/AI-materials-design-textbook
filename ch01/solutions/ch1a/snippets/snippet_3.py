import numpy as np
R = np.array([10,30,50,70,90.]); V = np.array([20.0,52.1,84.6,118.3,151.0]); N = 5
A = np.column_stack([R, np.ones(N)]); xs = np.linalg.lstsq(A, V, rcond=None)[0]
H = 2/N*A.T@A; g = lambda x: 2/N*A.T@(A@x - V)
def run(step, tol=1e-3, maxit=10**6):
    x, s = np.array([10., 10.]), {}
    for k in range(1, maxit+1):
        x = step(x, g(x), s, k)
        if not np.all(np.isfinite(x)): return "div"
        if np.linalg.norm(x - xs) < tol: return k
    return ">maxit"
def adam(lr, b1=.9, b2=.999, eps=1e-8):
    def st(x, gr, s, k):
        s["m"] = b1*s.get("m", 0) + (1-b1)*gr; s["v"] = b2*s.get("v", 0) + (1-b2)*gr*gr
        return x - lr*(s["m"]/(1-b1**k))/(np.sqrt(s["v"]/(1-b2**k)) + eps)
    return st
L, mu = np.linalg.eigvalsh(H)[::-1]; k = L/mu
hb_a, hb_b = 4/(np.sqrt(L)+np.sqrt(mu))**2, ((np.sqrt(k)-1)/(np.sqrt(k)+1))**2
def hb(x, gr, s, kk): s["v"] = hb_b*s.get("v", 0) + gr; return x - hb_a*s["v"]
print(run(lambda x, gr, s, kk: x - 3.02e-4*gr), run(hb), run(adam(0.3)),
      run(lambda x, gr, s, kk: x - np.linalg.solve(H, gr)))
