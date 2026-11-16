# 학생 도전 과제 (1.2.4-Why): f(x)=sqrt(x^2+1)-x 의 점근 거동과 x=1e8 값
import numpy as np, mpmath as mp
mp.mp.dps = 50
x = mp.mpf(10)**8
exact = mp.sqrt(x**2+1) - x
asym1 = 1/(2*x); asym2 = 1/(2*x) - 1/(8*x**3)
eps = mp.mpf(2)**-52
print("exact          =", mp.nstr(exact, 25))
print("1/(2x)         =", mp.nstr(asym1, 25))
print("1/(2x)-1/(8x^3)=", mp.nstr(asym2, 25))
print("rel correction 1/(4x^2) =", mp.nstr(1/(4*x**2), 6), " in units of eps:", mp.nstr(1/(4*x**2)/eps, 6))
print("exact / eps    =", mp.nstr(exact/eps, 12))
print("FP64 robust    =", repr(1.0/(np.sqrt(1e8**2+1)+1e8)), " == 5e-9 ?", 1.0/(np.sqrt(1e8**2+1)+1e8) == 5e-9)
print("rel err of FP64 robust vs exact:", mp.nstr(abs(mp.mpf(1.0/(np.sqrt(1e8**2+1)+1e8))-exact)/exact, 6))
print("naive: x^2+1 == x^2 in FP64?", (1e8**2+1) == 1e8**2, " spacing at 1e16:", np.spacing(1e16))
print("x where naive starts to fail ~ eps^-1/2 =", float(eps**-0.5))
