"""Cross-check for lab_5_10.jl: Onsager energy with K(m) from the AGM (as in the Julia file) vs scipy ellipk,
and Yang magnetization, at the temperatures of the chapter's correctness tables."""
import math
from scipy.special import ellipk
TC = 2/math.log(1+math.sqrt(2))
def K_agm(m):
    a, b = 1.0, math.sqrt(1-m)
    while abs(a-b) > 1e-15*a: a, b = (a+b)/2, math.sqrt(a*b)
    return math.pi/(2*a)
def e_ons(T, K):
    b = 1/T; k = 2*math.sinh(2*b)/math.cosh(2*b)**2
    return -(1/math.tanh(2*b))*(1+(2/math.pi)*(2*math.tanh(2*b)**2-1)*K(k*k))
for T in (1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
    y = (1-math.sinh(2/T)**-4)**0.125 if T < TC else 0.0
    print(f"T={T:.2f}: AGM {e_ons(T, K_agm):+.4f}  scipy {e_ons(T, ellipk):+.4f}  diff {abs(e_ons(T,K_agm)-e_ons(T,ellipk)):.1e}  Yang {y:.4f}")
