# 연습문제 1.5.E: 벤젠 Hückel pi-Hamiltonian
import numpy as np
n = 6
Adj = np.zeros((n, n))
for i in range(n): Adj[i, (i+1) % n] = Adj[(i+1) % n, i] = 1
# H = alpha*I + beta*Adj; 단위: alpha=0, beta=-1 로 놓으면 고유값 = alpha + x*beta 의 x
x = np.linalg.eigvalsh(Adj)[::-1]            # beta<0 이므로 x가 클수록 에너지 낮음
print("H = alpha I + beta * Adj, eigenvalues alpha + x beta, x =", np.round(x, 10))
print("analytic x_k = 2cos(2 pi k/6):", np.round(sorted(2*np.cos(2*np.pi*np.arange(6)/6), reverse=True), 10))
occ = x[:3]; print("occupied (6 e-, 2 per level): x =", occ, "-> E_pi = 6 alpha + %g beta" % (2*occ.sum()))
print("HOMO x =", x[2], " LUMO x =", x[3], " gap = %g |beta|" % (x[2]-x[3]))
print("delocalization energy vs 3 ethylenes (6a+6b): %g beta" % (2*occ.sum() - 6))
hc = 1239.84198  # eV nm
exp = {"1B2u": 4.90, "1B1u": 6.20, "1E1u": 6.94}   # Hirao, Nakano, Hashimoto, CPL 235, 430 (1995) 에 인용된 실험값
for k, E in exp.items(): print(f"{k}: {E} eV = {hc/E:.0f} nm, implied |beta| = E/2 = {E/2:.2f} eV")
# 예: beta = -2.45 eV 와 -3.47 eV 일 때 스펙트럼
for b in [2.45, 3.47]: print(f"|beta|={b}: levels (alpha=0) =", np.round(-x*b, 3), " HOMO-LUMO =", 2*b, "eV ->", round(hc/(2*b)), "nm")
# 고유벡터: HOMO/LUMO 이중 축퇴 확인
w, U = np.linalg.eigh(-Adj); print("degeneracies (beta=-1 units):", np.round(w, 6))
