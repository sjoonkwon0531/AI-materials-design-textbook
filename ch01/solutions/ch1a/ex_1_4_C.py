# 연습문제 1.4.C: rotameter 비용함수 Hessian 고유값과 학습률별 GD 거동
import numpy as np
R = np.array([10, 30, 50, 70, 90.]); V = np.array([20.0, 52.1, 84.6, 118.3, 151.0]); N = 5
A = np.column_stack([R, np.ones(N)])
H = 2/N * A.T @ A
lam = np.linalg.eigvalsh(H)
print("H = 2/N A^T A =\n", H)
print(f"eig(H) = {lam}, kappa(H) = {lam[1]/lam[0]:.2f}, kappa(A) = {np.linalg.cond(A):.3f}, kappa(A)^2 = {np.linalg.cond(A)**2:.2f}")
print(f"tr = {np.trace(H)}, det = {np.linalg.det(H):.4f}")
print(f"alpha_max = 2/lmax = {2/lam[1]:.4e}")
xstar = np.linalg.lstsq(A, V, rcond=None)[0]; print("x* =", xstar)
def gd(alpha, iters, x0=(10., 10.)):
    x = np.array(x0); hist = []
    for k in range(iters):
        x = x - alpha * 2/N * A.T @ (A @ x - V)
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e300: return x, k+1, hist
        hist.append(x.copy())
    return x, iters, hist
for alpha in [1e-3, 1e-4, 1e-5]:
    rho_fast = abs(1 - alpha*lam[1]); rho_slow = abs(1 - alpha*lam[0])
    print(f"\nalpha={alpha:g}: |1-a*lmax|={rho_fast:.6f}, |1-a*lmin|={rho_slow:.8f}")
    if rho_fast > 1:
        x, k, h = gd(alpha, 200)
        for kk in [10, 50, 100, 200]:
            if kk <= len(h): print(f"   iter {kk}: |x - x*| = {np.linalg.norm(h[kk-1]-xstar):.3e}")
    else:
        need = np.log(1e-6) / np.log(rho_slow)
        print(f"   predicted iters to shrink slow-mode error by 1e-6: {need:.3e}")
        for iters in [1000, 10000, 100000]:
            x, k, _ = gd(alpha, iters)
            print(f"   {iters:>6d} iters: a={x[0]:.4f}, b={x[1]:.4f}, |x-x*|={np.linalg.norm(x-xstar):.3e}")
# 고유벡터: 느린 방향은 거의 절편(b) 방향
w, Q = np.linalg.eigh(H); print("\neigvecs (cols: slow, fast):\n", Q)
