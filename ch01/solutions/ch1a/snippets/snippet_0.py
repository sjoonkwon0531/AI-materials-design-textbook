import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(0); f = lambda x: np.sin(2*np.pi*x)
xt = np.linspace(0, 1, 200); R, N, s = 100, 30, 0.1
data = [(x, f(x) + rng.normal(0, s, N)) for x in (rng.uniform(0, 1, N) for _ in range(R))]
for d in [1, 3, 5, 9, 15]:
    P = np.array([make_pipeline(PolynomialFeatures(d), LinearRegression())
                  .fit(x[:, None], y).predict(xt[:, None]) for x, y in data])
    b2, var = np.mean((P.mean(0) - f(xt))**2), np.mean(P.var(0))
    print(d, b2, b2 - var/(R-1), var, b2 + var + s**2)
