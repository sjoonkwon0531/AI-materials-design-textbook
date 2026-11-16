import numpy as np

R = np.array([10, 30, 50, 70, 90])
V = np.array([20.0, 52.1, 84.6, 118.3, 151.0])
N = len(R)

# Normal equation
A_mat = np.column_stack([R, np.ones(N)])
B_vec = V
X_vec = np.linalg.inv(A_mat.T @ A_mat) @ A_mat.T @ B_vec

a, b = X_vec
print(f"방법 1 (정규방정식): a = {a:.4f}, b = {b:.4f}")
# 출력: a = 1.6410, b = 3.1500
