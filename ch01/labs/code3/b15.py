class Dual:
    """Forward-mode AD via dual numbers."""
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)
    
    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real * other.real, 
                    self.real * other.dual + self.dual * other.real)
    
    def __repr__(self):
        return f"{self.real} + {self.dual}ε"

import math
def sin(x):
    if isinstance(x, Dual):
        return Dual(math.sin(x.real), x.dual * math.cos(x.real))
    return math.sin(x)

def cos(x):
    if isinstance(x, Dual):
        return Dual(math.cos(x.real), -x.dual * math.sin(x.real))
    return math.cos(x)

# 사용: f(x) = x^2 * sin(x), 점 x = 1에서의 미분
x = Dual(1.0, 1.0)  # x = 1, dx = 1
f = x * x * sin(x)
print(f)
# 출력: 0.8414709848078965 + 2.2232442754839328ε
# real = f(1) = sin(1) ≈ 0.8415
# dual = f'(1) = 2*sin(1) + cos(1) ≈ 2.2232
