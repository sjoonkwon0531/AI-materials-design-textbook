import numpy as np
from scipy.integrate import solve_ivp
def g(s):
    ev = lambda x, Y: abs(Y[0]) - 1e8; ev.terminal = True
    r = solve_ivp(lambda x, Y: [Y[1], Y[0]**2], (0, 1), [0, s], rtol=1e-11, atol=1e-12, events=ev)
    return r.y[0, -1] - 1 if r.status == 0 else np.inf
ss = np.concatenate([-np.logspace(3, 0, 400), np.linspace(-1, 5, 100)])
G = np.array([g(s) for s in ss])
fin = np.isfinite(G)
print("blow-up (u>1e8 before x=1) for s <", ss[~fin].max() if (~fin).any() else None, " count", (~fin).sum())
idx = np.where(fin[:-1] & fin[1:] & (np.sign(G[:-1]) != np.sign(G[1:])))[0]
for i in idx: print("sign change between", ss[i], ss[i+1], G[i], G[i+1])
