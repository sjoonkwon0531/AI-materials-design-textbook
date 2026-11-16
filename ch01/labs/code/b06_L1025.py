from scipy.optimize import minimize

def cost(params, x, y):
    a, b = params
    return np.mean((y - a*x - b)**2)

result = minimize(cost, x0=[1.0, 1.0], args=(R, V), method='Nelder-Mead')
a2, b2 = result.x
print(f"방법 2 (Nelder-Mead): a = {a2:.4f}, b = {b2:.4f}")
# 출력: a = 1.6400, b = 3.1500