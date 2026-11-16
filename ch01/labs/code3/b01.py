import numpy as np

x = 1e8
naive = np.sqrt(x**2 + 1) - x
print(f"Naive:  {naive:.16e}")
# 출력: Naive:  0.0000000000000000e+00  ← 완전히 0이 됨!
