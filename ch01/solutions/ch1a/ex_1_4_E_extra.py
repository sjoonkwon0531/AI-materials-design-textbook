# 1.4.E 보충: momentum beta=0.9에서 더 큰 학습률 격자, 그리고 1.4.B의 curve_fit 함수 평가 수
import numpy as np
from scipy.optimize import curve_fit
src = open("ex_1_4_E.py").read(); exec(src.split("results = {}")[0])
print(f"heavy-ball stability limit for beta=0.9: alpha < 2(1+beta)/L = {2*1.9/L:.3e}")
for lr in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 5.5e-4]:
    k, x, _ = run(momentum(lr, 0.9)); print(f"momentum beta=0.9 lr={lr:g}: {k}")
R_ = np.array([10, 30, 50, 70, 90.]); V_ = np.array([20.0, 52.1, 84.6, 118.3, 151.0])
p, cov, info, msg, ier = curve_fit(lambda R, a, b, c: a*R**b + c, R_, V_, p0=[1, 1, 0], full_output=True)
print("curve_fit nfev =", info["nfev"])
