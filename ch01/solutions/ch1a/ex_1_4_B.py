# 연습문제 1.4.B: V = a R^b + c 비선형 적합
import numpy as np
from scipy.optimize import curve_fit
R = np.array([10, 30, 50, 70, 90.]); V = np.array([20.0, 52.1, 84.6, 118.3, 151.0]); N = 5
model = lambda R, a, b, c: a * R**b + c
# (c) curve_fit (Levenberg-Marquardt)
p, cov = curve_fit(model, R, V, p0=[1, 1, 0])
rms = np.sqrt(np.mean((model(R, *p) - V)**2))
print(f"curve_fit: a={p[0]:.5f} b={p[1]:.5f} c={p[2]:.5f}  RMS={rms:.4f}  SSE={np.sum((model(R,*p)-V)**2):.5f}")
print(f"  J* = SSE/N = {np.mean((model(R,*p)-V)**2):.5f}")
print("  param std:", np.sqrt(np.diag(cov)))
print("  corr matrix:\n", np.round(cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov))), 4))
lin = np.polyfit(R, V, 1); print(f"linear fit SSE={np.sum((np.polyval(lin,R)-V)**2):.5f}")
# variable projection: b 고정 시 (a,c)는 선형 최소제곱 → 1차원 프로파일
def profile(b):
    A = np.column_stack([R**b, np.ones(N)]); x, *_ = np.linalg.lstsq(A, V, rcond=None)
    return np.sum((A @ x - V)**2), x
bs = np.linspace(0.8, 1.3, 5001); sse = np.array([profile(b)[0] for b in bs])
bo = bs[sse.argmin()]; print(f"profile SSE min at b={bo:.4f}, (a,c)={profile(bo)[1]}, SSE={sse.min():.5f}")
# Hessian of J=(1/N)SSE at optimum: conditioning
def J(q): return np.mean((model(R, *q) - V)**2)
def grad(q):
    a, b, c = q; r = model(R, a, b, c) - V; Rb = R**b
    return 2/N * np.array([np.sum(r*Rb), np.sum(r*a*Rb*np.log(R)), np.sum(r)])
Jac = np.column_stack([R**p[1], p[0]*R**p[1]*np.log(R), np.ones(N)])
Hgn = 2/N * Jac.T @ Jac; ev = np.linalg.eigvalsh(Hgn)
print(f"Gauss-Newton Hessian eig at optimum: {ev}, kappa={ev[-1]/ev[0]:.3e}, alpha_max=2/lmax={2/ev[-1]:.3e}")
# (b) 경사 하강법: 여러 초기값
def gd(q0, alpha, iters):
    q = np.array(q0, float)
    for k in range(iters):
        g = grad(q); qn = q - alpha*g
        if not np.all(np.isfinite(qn)) or J(qn) > 1e12: return q, k, "diverged"
        q = qn
    return q, iters, "ok"
alpha = 1e-6
for q0 in [(1, 1, 0), (2, 0.9, 0), (0.5, 1.2, 10), (10, 0.5, 0), (1, 1.5, 0)]:
    for iters in [10**5, 10**6]:
        q, k, st = gd(q0, alpha, iters)
        print(f"GD alpha={alpha:g} init={q0}: iters={k} status={st}  a={q[0]:.4f} b={q[1]:.4f} c={q[2]:.3f}  J={J(q):.4f}")
