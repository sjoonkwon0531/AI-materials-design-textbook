# 연습문제 1.4.A: [[3,1],[1,2]]의 LU와 Cholesky
import numpy as np, scipy.linalg as sl
A = np.array([[3., 1.], [1., 2.]])
L = np.array([[1, 0], [1/3, 1]]); U = np.array([[3, 1], [0, 5/3]])
print("hand LU ok:", np.allclose(L @ U, A))
P, Ls, Us = sl.lu(A); print("scipy P=\n", P, "\nL=\n", Ls, "\nU=\n", Us)
Lc = np.array([[np.sqrt(3), 0], [1/np.sqrt(3), np.sqrt(5/3)]])
print("hand chol:\n", Lc, "\nnumpy chol:\n", np.linalg.cholesky(A), "\nmatch:", np.allclose(Lc, np.linalg.cholesky(A)))
D = np.diag(np.diag(U)); print("LDL^T: D =", np.diag(D), " L D^(1/2) == chol:", np.allclose(L @ np.sqrt(D), Lc))
print("det = 3*5/3 =", np.prod(np.diag(U)), " eig =", np.linalg.eigvalsh(A))
