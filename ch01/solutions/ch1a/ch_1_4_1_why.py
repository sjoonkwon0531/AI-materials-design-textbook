# ★Why 1.4.1: row picture와 column picture가 같은 Ax를 준다 (분리기 예제)
import numpy as np
A = np.array([[1., 1.], [-2., 1.]]); b = np.array([3., 0.])
x = np.linalg.solve(A, b); print("x =", x)
row = np.array([A[i] @ x for i in range(2)]); col = x[0]*A[:, 0] + x[1]*A[:, 1]
print("row picture (dot products):", row, " column picture (combination):", col)
