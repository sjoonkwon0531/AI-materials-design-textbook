# 1.3.E 후처리: ex_1_3_E.py가 저장한 결과로 Var/R 보정된 Bias^2 계산
import numpy as np
d = np.load("ex_1_3_E.npz"); xt, P64, P32 = d["xt"], d["P64"], d["P32"]; R = P64.shape[0]
f = lambda x: np.sin(5*x)*np.cos(x**2)
var = P64.var(0, ddof=1); b2 = (P64.mean(0)-f(xt))**2 - var/R; prec = ((P32-P64)**2).mean(0)
for k, m in {"|x|<1": np.abs(xt)<1, "1<=|x|<1.75": (np.abs(xt)>=1)&(np.abs(xt)<1.75), "|x|>=1.75": np.abs(xt)>=1.75}.items():
    print(f"{k:12s}: Bias^2(corr)={b2[m].mean():.2e}  Var={var[m].mean():.2e}  Prec={prec[m].mean():.2e}  Prec/Var={prec[m].mean()/var[m].mean():.3f}")
print(f"fraction of x where Prec > Var: {np.mean(prec > var):.3f};  where Bias^2(corr) > Var: {np.mean(b2 > var):.3f}")
# 같은 FP32 데이터로 '반올림 수준' 오차와의 비교: FP32 eps * 출력 크기
print(f"FP32 eps*max|f| = {np.finfo(np.float32).eps*np.abs(f(xt)).max():.2e}")
