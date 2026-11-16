robust = 1.0 / (np.sqrt(x**2 + 1) + x)
print(f"Robust: {robust:.16e}")
# 출력: Robust: 5.0000000000000000e-09  ← 정확한 답!
