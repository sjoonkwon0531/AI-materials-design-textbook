# 연습문제 1.5.B: 첫 주성분 = 분산 최대 방향 (Rayleigh 몫) 수치 확인
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3)) @ np.array([[3, 0, 0], [1, 1, 0], [0.5, 0.2, 0.3]])
Xc = X - X.mean(0); C = Xc.T @ Xc / (len(X) - 1)
w, Q = np.linalg.eigh(C)
best = 0
for _ in range(200000 // 1000):
    U = rng.normal(size=(1000, 3)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    best = max(best, np.max(np.var(Xc @ U.T, axis=0, ddof=1)))
print(f"lambda_max(C) = {w[-1]:.6f}, max sampled projected variance = {best:.6f}, var along u1 = {np.var(Xc @ Q[:, -1], ddof=1):.6f}")
