import math
b1, b2, eta, lam = 0.9, 0.999, 1e-3, 0.5
th, m, v = 1.0, 0.0, 0.0
for t in range(1, 7001):
    g = lam*th; m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g*g
    th -= eta*(m/(1-b1**t))/(math.sqrt(v/(1-b2**t))+1e-8)
    if t in (6000, 6200, 6216, 6217, 6218, 6300, 7000): print(t, th)
# simple model: sqrt(v_hat) ~ lam * rms of theta over history (uniform average), linear theta
for t in (100, 500):
    rms = math.sqrt(sum((1-eta*s)**2 for s in range(t))/t); print("t", t, "linear-model ratio", (1-eta*t)/rms)
