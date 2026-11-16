# ★Why 1.3.4 도전 과제: Fbar 대신 임의의 g를 쓰면 cross term은?
import numpy as np
rng = np.random.default_rng(1)
Fstar = 1.2
Fhat = 0.7 + rng.exponential(0.3, size=2_000_000)   # 비대칭(skewed) 추정량 분포
mse = np.mean((Fstar - Fhat)**2)
for name, g in [("mean", Fhat.mean()), ("median", np.median(Fhat)), ("F*", Fstar), ("0.5", 0.5)]:
    t1 = (Fstar - g)**2; t2 = np.mean((g - Fhat)**2); cross = 2*(Fstar - g)*(g - Fhat.mean())
    print(f"g={name:6s}: g={g:.4f}  (F*-g)^2={t1:.5f}  E(g-F)^2={t2:.5f}  cross={cross:+.5f}  sum={t1+t2+cross:.5f}  MSE={mse:.5f}")
