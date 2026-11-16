import numpy as np
roots = np.array([1.0, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)])
N, span, max_iter, tol = 1000, 2.0, 50, 1e-6
x = np.linspace(-span, span, N); X, Y = np.meshgrid(x, x)
Zk = X + 1j*Y
basin  = np.full(Zk.shape, -1, dtype=int)
itc    = np.zeros(Zk.shape, dtype=int)
active = np.ones(Zk.shape, dtype=bool)
for k in range(max_iter):
    df = 3.0*Zk**2; safe = np.abs(df) > 1e-12
    step = np.zeros_like(Zk); step[safe] = (Zk[safe]**3 - 1.0)/df[safe]
    Zk = Zk - step
    for ri, r in enumerate(roots):
        hit = active & (np.abs(Zk - r) < tol)
        basin[hit] = ri; itc[hit] = k + 1; active[hit] = False
    if not active.any(): break
print("converged fraction =", np.mean(basin >= 0))      # 1.0000
print("mean iterations    =", np.mean(itc[basin >= 0])) # 6.843
print("basin fractions    =", [round(np.mean(basin==i),4) for i in range(3)])
# -> [0.3528, 0.3236, 0.3236]