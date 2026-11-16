# 연습문제 1.3.D: 다항 회귀 bias-variance 실험
# 해석: x ~ U(0,1), 학습점 N=30, sigma=0.1, 100개 학습셋, 평가격자 200점(x in [0,1])
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
Ntr, sigma, R = 30, 0.1, 100
f = lambda x: np.sin(2*np.pi*x)
xt = np.linspace(0, 1, 200)
degs = [1, 3, 5, 9, 15]
datasets = []
for r in range(R):
    x = rng.uniform(0, 1, Ntr); y = f(x) + rng.normal(0, sigma, Ntr)
    datasets.append((x, y))
rows = []
for d in degs:
    preds = np.empty((R, xt.size))
    for r, (x, y) in enumerate(datasets):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], y)
        preds[r] = m.predict(xt[:, None])
    Fbar = preds.mean(0)
    bias2 = np.mean((Fbar - f(xt))**2)
    var = np.mean(preds.var(0))
    var_med = np.median(preds.var(0))
    rows.append((d, bias2, var, bias2 + var + sigma**2, var_med))
    b2c = bias2 - var/(R-1)   # Fbar 추정의 유한표본 오염(Var/R) 보정
    print(f"d={d:2d}: Bias^2(corr)={b2c:.3e}  Bias^2={bias2:.3e}  Var={var:.3e}  Total(+sigma^2)={bias2+var+sigma**2:.3e}   median_x Var={var_med:.3e}")
# 경계 효과: d=15 분산의 위치 의존성
d = 15
preds = np.array([make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], y).predict(xt[:, None]) for x, y in datasets])
v = preds.var(0)
inner = (xt > 0.1) & (xt < 0.9)
print(f"d=15 Var: inner [0.1,0.9] mean={v[inner].mean():.3e}, edges mean={v[~inner].mean():.3e}")
# 동일 실험 반복 (seed 다르게)으로 결론의 강건성 확인
for seed in [1, 2]:
    rng2 = np.random.default_rng(seed); out = []
    ds = [(lambda x: (x, f(x) + rng2.normal(0, sigma, Ntr)))(rng2.uniform(0, 1, Ntr)) for _ in range(R)]
    for d in degs:
        P = np.array([make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], y).predict(xt[:, None]) for x, y in ds])
        out.append(np.mean((P.mean(0)-f(xt))**2) + np.mean(P.var(0)) + sigma**2)
    print(f"seed {seed}: total =", ["%.3e" % t for t in out], " argmin d =", degs[int(np.argmin(out))])
r = np.array(rows)
plt.semilogy(r[:,0], r[:,1], "o-", label="Bias$^2$"); plt.semilogy(r[:,0], r[:,2], "s-", label="Variance")
plt.semilogy(r[:,0], r[:,3], "^-", label="Total"); plt.axhline(sigma**2, ls="--", c="gray")
plt.xlabel("degree d"); plt.legend(); plt.savefig("ex_1_3_D.png", dpi=120)
# 설계행렬 조건수(수치 오차와의 연결, §1.3.5)
x0, _ = datasets[0]
for d in [3, 5, 9, 15]:
    print(f"d={d:2d}: cond(Vandermonde, N=30) = {np.linalg.cond(PolynomialFeatures(d).fit_transform(x0[:,None])):.2e}")
# d=15 분산의 heavy tail: 학습셋별 평가격자 최대 |예측|
mx = np.abs(preds).max(1)
print("d=15 per-dataset max|pred| percentiles (50,90,99):", np.percentile(mx, [50, 90, 99]).round(2))
