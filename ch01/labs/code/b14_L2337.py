import sympy as sp
x = sp.Symbol('x')
f = x**2 * sp.sin(x)
print(sp.diff(f, x))
# 출력: 2*x*sin(x) + x**2*cos(x)