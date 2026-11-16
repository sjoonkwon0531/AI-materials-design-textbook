# ★Why 1.2.7: 단위벡터 (sqrt2/2, sqrt2/2)의 세 norm
import numpy as np
x = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
for p in [1, 2, np.inf]:
    print(f"||x||_{p} = {np.linalg.norm(x, p):.6f}")
# 일반 부등식 ||x||_inf <= ||x||_2 <= ||x||_1 <= sqrt(n)||x||_2 랜덤 검증
rng = np.random.default_rng(0); ok = True
for _ in range(10000):
    v = rng.normal(size=5)
    n1, n2, ni = (np.linalg.norm(v, p) for p in (1, 2, np.inf))
    ok &= (ni <= n2 + 1e-15) and (n2 <= n1 + 1e-15) and (n1 <= np.sqrt(5)*n2 + 1e-12)
print("inequalities hold on 10000 random vectors:", ok)
