# 연습문제 1.3.B: A=[[4.1,2.8],[9.7,6.6]]의 1, 2, inf-norm 조건수
import numpy as np
from fractions import Fraction as F
A = np.array([[4.1, 2.8], [9.7, 6.6]])
Af = [[F("4.1"), F("2.8")], [F("9.7"), F("6.6")]]
det = Af[0][0]*Af[1][1] - Af[0][1]*Af[1][0]
Ainv = [[Af[1][1]/det, -Af[0][1]/det], [-Af[1][0]/det, Af[0][0]/det]]
print("det =", det, "=", float(det)); print("A^-1 =", [[str(v) for v in r] for r in Ainv])
col = lambda M: max(abs(M[0][j]) + abs(M[1][j]) for j in range(2))
row = lambda M: max(abs(M[i][0]) + abs(M[i][1]) for i in range(2))
print("||A||_1 =", float(col(Af)), " ||A^-1||_1 =", float(col(Ainv)), " kappa_1 =", float(col(Af)*col(Ainv)))
print("||A||_inf =", float(row(Af)), " ||A^-1||_inf =", float(row(Ainv)), " kappa_inf =", float(row(Af)*row(Ainv)))
ATA = A.T @ A; lam = np.linalg.eigvalsh(ATA)
print("eig(A^T A) =", lam, " sigma =", np.sqrt(lam))
print("kappa_2 = sqrt(lmax/lmin) =", np.sqrt(lam[1]/lam[0]))
# 2x2 닫힌 형식: kappa_2 = (s + sqrt(s^2-4d^2))/(2|d|), s=||A||_F^2
s = (A**2).sum(); d = abs(np.linalg.det(A))
print("||A||_F^2 =", s, " closed form kappa_2 =", (s + np.sqrt(s*s - 4*d*d))/(2*d), " approx s/|d| =", s/d)
for p in [1, 2, np.inf]:
    print(f"np.linalg.cond(A, p={p}) = {np.linalg.cond(A, p):.6f}")
