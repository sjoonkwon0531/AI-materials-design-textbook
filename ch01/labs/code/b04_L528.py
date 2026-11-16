import numpy as np

# 원래 행렬과 정확한 역행렬
A = np.array([[4.1, 2.8], [9.7, 6.6]])
A_inv_exact = np.array([[-66.0, 28.0], [97.0, -41.0]])

# 수치적으로 계산한 역행렬
A_inv_numerical = np.linalg.inv(A)
print("Numerical A^(-1):")
print(A_inv_numerical)
# 출력 예:
# [[-66.00000000000006   28.000000000000025]
#  [ 97.00000000000009  -41.000000000000036]]

# 조건수와 예측되는 자릿수 손실
cond = np.linalg.cond(A)
print(f"\nκ(A) = {cond:.4e}")
print(f"Expected lost digits: log10(κ) ≈ {np.log10(cond):.2f}")

# 작은 perturbation을 주면?
A_perturbed = A + np.array([[0.0, 0.0], [-0.024, 0.008]])
print(f"\nperturbed A:")
print(A_perturbed)
print(f"\nperturbed A^(-1):")
print(np.linalg.inv(A_perturbed))  # 결과가 완전히 다름!