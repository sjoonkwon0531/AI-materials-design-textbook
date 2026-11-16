import cvxpy as cp

X = cp.Variable(2)
objective = cp.Minimize(cp.norm(A_mat @ X - B_vec, 2))
problem = cp.Problem(objective)
problem.solve()

a3, b3 = X.value
print(f"방법 3 (CVXPY): a = {a3:.4f}, b = {b3:.4f}")
# 출력: a = 1.6410, b = 3.1500
