"""
Lab 1.8: 같은 함수 f(x) = x² sin(x)를 4가지로 미분하여 비교한다.
"""
import math
import numpy as np
import sympy as sp
import torch

# === 방법 1: 손으로 구한 해석적 미분 ===
def f_analytical_derivative(x):
    return 2*x*math.sin(x) + x**2 * math.cos(x)

# === 방법 2: 기호 미분 (SymPy) ===
x_sym = sp.Symbol('x')
f_sym = x_sym**2 * sp.sin(x_sym)
df_sym = sp.diff(f_sym, x_sym)
f_symbolic = sp.lambdify(x_sym, df_sym, 'math')

# === 방법 3: 수치 미분 (중심차분) ===
def f(x):
    return x**2 * math.sin(x)

def f_numerical_derivative(x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)

# === 방법 4: 자동미분 (PyTorch) ===
def f_torch_derivative(x_value):
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)  # FP64!
    y = x**2 * torch.sin(x)
    y.backward()
    return x.grad.item()

# === 비교 ===
x0 = 1.0
exact = 2*1.0*math.sin(1.0) + 1.0**2 * math.cos(1.0)

methods = {
    'Analytical (by hand)': f_analytical_derivative(x0),
    'Symbolic (SymPy)':     f_symbolic(x0),
    'Numerical (h=1e-5)':   f_numerical_derivative(x0, 1e-5),
    'Numerical (h=1e-10)':  f_numerical_derivative(x0, 1e-10),
    'Automatic (PyTorch)':  f_torch_derivative(x0),
}

print(f"{'Method':25s} | {'Value':>20s} | {'|Error|':>10s}")
print("-" * 65)
for name, value in methods.items():
    print(f"{name:25s} | {value:20.16f} | {abs(value - exact):10.2e}")

# 출력:
# Analytical (by hand)      |   2.2232442754839328 |   0.00e+00
# Symbolic (SymPy)          |   2.2232442754839328 |   0.00e+00
# Numerical (h=1e-5)        |   2.2232442754455128 |   3.84e-11
# Numerical (h=1e-10)       |   2.2232438112723685 |   4.64e-07
# Automatic (PyTorch)       |   2.2232442754839328 |   0.00e+00
