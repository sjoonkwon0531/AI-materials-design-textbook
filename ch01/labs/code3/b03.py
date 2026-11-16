# Lab 1.2: Floating-point precision in action
# Goal: 4가지 정밀도에서 동일 계산을 비교

import numpy as np
import torch

# 1. 머신 엡실론 직접 측정
def find_machine_epsilon(dtype):
    """1.0 + eps > 1.0이 성립하는 가장 작은 eps를 찾는다."""
    eps = dtype(1.0)
    while dtype(1.0) + eps / dtype(2.0) > dtype(1.0):
        eps = eps / dtype(2.0)
    return eps

print("--- Machine epsilon (measured) ---")
print(f"FP64: {find_machine_epsilon(np.float64):.4e}")
print(f"FP32: {find_machine_epsilon(np.float32):.4e}")
print(f"FP16: {find_machine_epsilon(np.float16):.4e}")
# BF16은 numpy에 없으므로 torch 사용
bf16_eps = torch.finfo(torch.bfloat16).eps
print(f"BF16: {bf16_eps:.4e}")

# 2. Catastrophic cancellation 체험
print("\n--- Catastrophic cancellation ---")
for x_val in [1e2, 1e4, 1e6, 1e8, 1e10]:
    x32 = np.float32(x_val)
    x64 = np.float64(x_val)
    naive32 = np.sqrt(x32**2 + 1) - x32
    naive64 = np.sqrt(x64**2 + 1) - x64
    robust = 1.0 / (np.sqrt(x64**2 + 1) + x64)
    print(f"x={x_val:.0e}: naive(FP32)={naive32:.4e}, "
          f"naive(FP64)={naive64:.4e}, robust(FP64)={robust:.4e}")

# 3. 누적 합 (associativity 위반)
print("\n--- Sum is not associative in FP ---")
small_values = np.full(10**7, 1e-8, dtype=np.float32)
total_a = np.float32(1.0) + small_values.sum()
total_b = small_values.sum() + np.float32(1.0)  # 순서 동일, 결과 동일
total_c = sum(small_values, np.float32(1.0))     # 하나씩 누적
print(f"Vectorized sum:    {total_a:.10f}")
print(f"Reverse order:     {total_b:.10f}")
print(f"Sequential sum:    {total_c:.10f}")
# 결과: 누적 순서에 따라 결과가 다름!
