X_gd = np.array([10.0, 10.0])
alpha = 1e-4      # 학습률 줄임
tol = 1e-10       # tolerance 줄임
max_iter = 100000  # iteration 늘림

for i in range(max_iter):
    pred = A_mat @ X_gd
    cost_old = np.mean((pred - V)**2)
    grad = 2 * A_mat.T @ (pred - V) / N
    X_gd = X_gd - alpha * grad
    pred_new = A_mat @ X_gd
    cost_new = np.mean((pred_new - V)**2)
    if abs(cost_old - cost_new) < tol:
        break

print(f"개선된 GD ({i+1} iter): a = {X_gd[0]:.4f}, b = {X_gd[1]:.4f}")
# 출력: a = 1.6400, b = 3.1500 ← 정상 도달!