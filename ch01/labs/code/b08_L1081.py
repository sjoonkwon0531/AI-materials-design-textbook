# Gradient Descent (matching Wk2 MATLAB code parameters)
X_gd = np.array([10.0, 10.0])  # 초기값
alpha = 2e-4                    # 학습률 (step size)
tol = 1e-4                      # 수렴 tolerance
max_iter = 1000

for i in range(max_iter):
    pred = A_mat @ X_gd
    cost_old = np.mean((pred - V)**2)
    grad = 2 * A_mat.T @ (pred - V) / N
    X_gd = X_gd - alpha * grad
    pred_new = A_mat @ X_gd
    cost_new = np.mean((pred_new - V)**2)
    if abs(cost_old - cost_new) < tol:
        break

a4, b4 = X_gd
print(f"방법 4 (GD, {i+1} iter): a = {a4:.4f}, b = {b4:.4f}")
# 출력: a = 1.6700, b = 1.2100  ← 다름!