# 연습문제 1.5.C: 1D 무한 우물 유한차분 (hbar=m=L=1), 내부 격자점 N=100
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
N, L = 100, 1.0; h = L/(N+1)
main = np.full(N, 1/h**2); off = np.full(N-1, -1/(2*h**2))      # H = -(1/2) d2/dx2
H = sp.diags([off, main, off], [-1, 0, 1], format="csc")
E = np.sort(eigsh(H, k=10, sigma=0, which="LM")[0])             # shift-invert (sigma=0)
n = np.arange(1, 11)
Eex = n**2*np.pi**2/2
Edisc = (2/h**2)*np.sin(n*np.pi*h/2)**2                        # 이산 해석해
print(" n   E_FD          E_exact       rel.err      E_disc(analytic FD)")
for i in range(10):
    print(f"{n[i]:2d}  {E[i]:12.6f}  {Eex[i]:12.6f}  {(E[i]-Eex[i])/Eex[i]:+.3e}   {Edisc[i]:12.6f}")
print("max |E_FD - E_disc| =", np.abs(E - Edisc).max())
print("rel.err / (n pi h)^2/12 :", ((Eex-E)/Eex / ((n*np.pi*h)**2/12)).round(4))
# 수렴 차수 확인 (n=1)
for NN in [50, 100, 200, 400]:
    hh = 1/(NN+1); e1 = (2/hh**2)*np.sin(np.pi*hh/2)**2
    print(f"N={NN}: rel.err(E1) = {(np.pi**2/2 - e1)/(np.pi**2/2):.3e}")
# eigsh without shift-invert (which='SA') 와 비교
E2 = np.sort(eigsh(H, k=10, which="SA", maxiter=100000)[0]); print("which='SA' agrees:", np.allclose(E, E2))
