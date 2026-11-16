import numpy as np

# 원래 행렬과 정확한 역행렬
A = np.array([[4.1, 2.8], [9.7, 6.6]])
A_inv_exact = np.array([[-66.0, 28.0], [97.0, -41.0]])

# 수치적으로 계산한 역행렬
A_inv_numerical = np.linalg.inv(A)
print("Numerical A^(-1):")
print(A_inv_numerical)
# 출력 (np.set_printoptions(precision=17)):
# [[-66.00000000000023  28.0000000000001 ]
#  [ 97.00000000000034 -41.00000000000014]]   # 상대오차 3.5e-15

# 조건수와 예측되는 자릿수 손실
cond = np.linalg.cond(A)
print(f"\nκ(A) = {cond:.4e}")
print(f"Expected lost digits: log10(κ) ≈ {np.log10(cond):.2f}")
# 출력: κ(A) = 1.6230e+03,  Expected lost digits: log10(κ) ≈ 3.21

# 작은 perturbation을 주면?
A_perturbed = A + np.array([[0.0, 0.0], [-0.024, 0.008]])
print(f"\nperturbed A:")
print(A_perturbed)
print(f"\nperturbed A^(-1):")
print(np.linalg.inv(A_perturbed))
# 출력: 원소가 ~1e15 수준인 행렬 (반올림 잡음이 그대로 드러남)

# 더 작은 perturbation 하나: A_22만 0.01 바꾸기
A_q = A + np.array([[0.0, 0.0], [0.0, 0.01]])
print(np.linalg.inv(A_q))
# 출력: [[-112.03   47.46]
#        [ 164.41  -69.49]]
